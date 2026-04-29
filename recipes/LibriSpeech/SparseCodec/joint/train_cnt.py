#!/usr/bin/env python3
"""Recipe for training a sparse codec-based ASR system on LibriSpeech
in a discriminative style. The system is using ASR downstream with CTC loss 
and a simple LSTM architecture. And a speaker recognition ECAPA head with AAM loss.


To run this recipe, do the following:
> python train.py hparams/hparams.yaml

Authors
 * Anastasia Kuznetsova 2026
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import pandas as pd
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
from speechbrain.nnet import loss
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main

logger = logging.getLogger(__name__)


def compute_speaker_embedding(wavs: torch.Tensor) -> torch.Tensor:
    """Compute speaker embeddings.
    Args:
        wavs: Tensor of shape (batch_size, num_samples) containing the input waveforms
    Returns:
        spk_emb: Tensor of shape (batch_size, embedding_dim) containing the speaker embeddings
    """
    with torch.no_grad():
        wavs = wavs.to(sparse_brain.device)

        codec_outputs = sparse_brain.modules.tokenizer(
            wavs.unsqueeze(0), n_quantizers=sparse_brain.hparams.num_codebooks
        )
        spk_emb = sparse_brain.modules.spk_encoder(codec_outputs[0].transpose(1, 2))
    return spk_emb


def compute_embedding_loop(
    data_loaders: List[torch.utils.data.DataLoader],
) -> Dict[str, torch.Tensor]:
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    Args:
        data_loader: DataLoader providing batches of waveforms and their corresponding IDs
    Returns:
        embedding_dict: Dictionary mapping segment IDs to their corresponding speaker embeddings
    """
    embedding_dict = {}

    with torch.no_grad():
        for data_loader in data_loaders:
            for batch in tqdm(data_loader, dynamic_ncols=True):
                seg_ids = [batch["id"]]
                wavs = batch["sig"]
                wavs = wavs.to(sparse_brain.device)
                emb = compute_speaker_embedding(wavs)

                for i, seg_id in enumerate(seg_ids):
                    if seg_id in embedding_dict:
                        continue
                    else:
                        embedding_dict[seg_id] = emb[i].detach().clone()
    return embedding_dict


# Define training procedure
class SparseBrain(sb.core.Brain):
    # def on_fit_batch_start(self, batch, should_step):
    #    """Freeze speaker branch for first 1000 steps, unfreeze after."""
    #    if not hasattr(self, "global_step"):
    #        self.global_step = 0
    #    # Freeze speaker branch for first 1000 steps
    #    if self.global_step < 1000:
    #        for p in self.modules.spk_encoder.parameters():
    #            p.requires_grad = False
    #        for p in self.modules.spk_classifier.parameters():
    #            p.requires_grad = False
    #    elif self.global_step == 1000:
    #        # Unfreeze at step 1000
    #        for p in self.modules.spk_encoder.parameters():
    #            p.requires_grad = True
    #        for p in self.modules.spk_classifier.parameters():
    #            p.requires_grad = True
    #    # Increment step counter
    #    self.global_step += 1
    #    super().on_fit_batch_start(batch, should_step)


    def compute_forward(self, batch: Dict[str, torch.Tensor], stage: sb.Stage) -> Tuple:
        """Forward computations from the waveform batches to the output probabilities.
        Computes the outputs for both the ASR head and the speaker classification head.
        Args:
            batch: Dictionary containing the input waveforms and their corresponding IDs
            stage: Current stage of the training process (TRAIN, VALID, TEST)
        Returns:
            Tuple containing the outputs for the ASR head and the speaker classification head
        """

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        encoder_out = self.modules.codec.encoder(wavs.unsqueeze(1))
        z_proj_content, z_proj_speaker, h,  sparse_loss, l1_reg_content, l1_reg_speaker = self.modules.disentangle(encoder_out)
        content_enc_input = self.modules.cnn(z_proj_content)

        # Top part of the in_tokens is used for ASR, and the bottom part is used for speaker classification
        enc_out, _ = self.modules.asr_encoder(
            content_enc_input, lengths=wav_lens
        )
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)


        # Compute ASR predictions (words) for validation and testing stages
        pred_hyps = None
        if stage == sb.Stage.VALID:
            pred_hyps = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            pred_hyps = test_searcher(p_ctc, wav_lens)

        # Speaker classification head forward pass
        # Bottom part of the in_tokens is used for speaker classification
        spk_emb = self.modules.spk_encoder(z_proj_speaker)
        spk_logits = self.modules.spk_classifier(spk_emb).squeeze(1)

        # Collect utterance embeddings for Cosine similarity evaluation
        if stage != sb.Stage.TRAIN:
            if not hasattr(self, "eval_spk_embs"):
                self.eval_spk_embs = {}
            utt_ids = batch.id
            for utt_id, spk_embedding in zip(utt_ids, spk_emb):
                self.eval_spk_embs[utt_id] = spk_embedding.cpu().detach()
        return (
            p_ctc,  # ctc probabilities
            pred_hyps,  # predicted hypotheses (token ids)
            wav_lens,
            spk_logits,
            sparse_loss,
            h,
            l1_reg_content,
            l1_reg_speaker
        )

    def compute_objectives(
        self, predictions: Tuple, batch: Dict[str, torch.Tensor], stage: sb.Stage
    ) -> torch.Tensor:
        """Computes combined ASR and Speaker verification loss.
        Stores individual loss components in a dictionary for logging and analysis.
        Args:
            predictions: Tuple containing the outputs from the forward pass
            batch: Dictionary containing the input waveforms and their corresponding IDs
            stage: Current stage of the training process (TRAIN, VALID, TEST)
        Returns:
            Combined loss as a torch.Tensor
        """
        # Unpack predictions from the forward() step.
        (   p_seq,  # ctc probabilities
            pred_hyps,  # predicted hypotheses (token ids)
            wav_lens,
            spk_logits,
            sparse_loss,
            h,
            l1_reg_content,
            l1_reg_speaker
        ) = predictions

        uttid = batch.id
        tokens, tokens_lens = batch.tokens
        spk_targets, _ = batch.spk_id_encoded

        ctc_batch_loss = self.hparams.ctc_cost(
            p_seq, tokens, wav_lens, tokens_lens, reduction=self.hparams.loss_reduction
        )
        ctc_batch_loss = ctc_batch_loss * self.hparams.ctc_weight
        sparse_batch_loss = sparse_loss * self.hparams.sparse_loss_weight

        # Speaker branch warmup: zero out speaker losses during warmup
        if stage == sb.Stage.TRAIN and hasattr(self, "global_step") and self.global_step <= 1000:
            batch_aam_loss = torch.tensor(0.0, device=ctc_batch_loss.device)
            spk_reg_loss = torch.tensor(0.0, device=ctc_batch_loss.device)
        else:
            if stage == sb.Stage.TRAIN:
                batch_aam_loss = self.hparams.spk_aam_loss(spk_logits, spk_targets)
                batch_aam_loss = batch_aam_loss * self.hparams.spk_aam_loss_weight
            else:
                batch_aam_loss = torch.tensor(0.0, device=ctc_batch_loss.device)
            spk_reg_loss = l1_reg_speaker * self.hparams.spk_reg_weight
        content_reg_loss = l1_reg_content * self.hparams.content_reg_weight

        loss = (
            ctc_batch_loss
            + sparse_batch_loss
            + batch_aam_loss
            + spk_reg_loss
            + content_reg_loss
        )
        # Decode words for ASR predictions
        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = self.tokenizer(pred_hyps, task="decode_from_list")
        elif stage == sb.Stage.TEST:
            predicted_words = [hyp[0].text.split(" ") for hyp in pred_hyps]

        # Compute WER and speaker verification error metrics for validation and testing stages
        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(uttid, predicted_words, target_words)
            spk_predictions = torch.argmax(spk_logits, dim=1)
            self.spk_error_metrics.append(
                uttid, spk_predictions, batch.spk_id_encoded.data
            )
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                # 1. Sparsity Percentage (How many are zero?)
                # predictions['h'] shape: [B, L, T, D]
                n_elements = h.numel()
                n_zero = (h.abs() < 1e-4).sum().float()
                sparsity_level = (n_zero / n_elements) * 100
                
                # 2. Active atoms per layer
                active_per_layer = (h.abs() > 1e-4).float().mean(dim=(0, 2, 3))
            # Log individual losses with file logger
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": self.hparams.epoch_counter.current},
                train_stats={
                    "loss": loss.item(),
                    "loss_ctc": ctc_batch_loss.item(),
                    "loss_sparse": sparse_batch_loss.item(),
                    "loss_aam": batch_aam_loss.item(),
                    "loss_spk_reg": spk_reg_loss.item(),
                    "loss_content_reg": content_reg_loss.item(),
                    "sparsity": sparsity_level.item(),
                    **{f"active_atoms_layer_{i}": active_per_layer[i].item() for i in range(len(active_per_layer))}
                },
                verbose=True,
                )

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()
        # NOTE: (anakuzne) change max_key to None for quantizer only inference
        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
            max_num_checkpoints=self.hparams.avg_checkpoints,
        )

        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.
        Initializes the WER and speaker verification error metrics for validation and testing stages.
        """
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.spk_error_metrics = self.hparams.spk_error_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""

        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            # Log average of individual loss components for the training stage
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["ErrorRate"] = self.spk_error_metrics.summarize("average")
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["ErrorRate"] = self.spk_error_metrics.summarize("average")

            # Compute cosine similarity stats for speaker verification
            # Load verification trial pairs
            veri_file = (
                self.hparams.valid_veri_file
                if stage == sb.Stage.VALID
                else self.hparams.test_veri_file
            )
            trial_pairs = load_verification_trials(veri_file)

            similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

            positive_scores = []
            negative_scores = []

            for label, utt1, utt2 in trial_pairs:
                spk_emb1 = self.eval_spk_embs.get(utt1)
                spk_emb2 = self.eval_spk_embs.get(utt2)
                if spk_emb1 is None or spk_emb2 is None:
                    logger.warning(f"Speaker embedding not found for {utt1} or {utt2}. Skipping this pair.")
                    continue
                cos_sim = similarity(spk_emb1, spk_emb2)
                if label == "1":
                    positive_scores.append(cos_sim.item())
                else:
                    negative_scores.append(cos_sim.item())
            positive_scores = torch.tensor(positive_scores)
            negative_scores = torch.tensor(negative_scores)
            eer, _ = self.hparams.spk_verification_metrics(
                positive_scores, negative_scores
            )
            stage_stats["EER"] = eer
            logger.info(
                "Statistics for epoch %s: EER %s, WER %s",
                str(epoch),
                stage_stats["EER"],
                stage_stats["WER"],
            )

        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.output_wer_folder, "w", encoding="utf-8") as w:
                    self.wer_metric.write_stats(w)

            self.checkpointer.save_and_keep_only(
                meta={"WER": 0.0, "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=1,
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if (
            should_step
            and type(self.hparams.scheduler).__name__ == "LinearNoamScheduler"
        ):
            self.hparams.scheduler(self.optimizer)


def load_verification_trials(veri_file: str) -> List[Tuple[str, str, str]]:
    """Load verification trial pairs from a file.
    Each line in the file should contain either:
    1. A label and two file paths (e.g., "1 path/to/file1.wav path/to/file2.wav")
    2. A comma-separated list of utterance IDs (no label)
    Returns a list of tuples containing the label (or None if not provided) and the two utterance IDs.
    """
    import os

    trial_pairs = []
    with open(veri_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label, path1, path2 = parts
                # Extract uttid from path (e.g., 422-122949-0032.wav -> 422-122949-0032)
                uttid1 = os.path.splitext(os.path.basename(path1))[0]
                uttid2 = os.path.splitext(os.path.basename(path2))[0]
                trial_pairs.append((label, uttid1, uttid2))
            else:
                # Try to parse as a comma-separated list of utterance IDs (no label)
                line_clean = (
                    line.strip().replace("[", "").replace("]", "").replace("'", "")
                )
                utt_ids = [utt.strip() for utt in line_clean.split(",") if utt.strip()]
                for i in range(0, len(utt_ids) - 1, 2):
                    trial_pairs.append((None, utt_ids[i], utt_ids[i + 1]))
    return trial_pairs


def dataio_prepare(hparams: dict, tokenizer: SentencePiece) -> Tuple:
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    Args:
        hparams: Dictionary containing the hyperparameters and configurations for data preparation
        tokenizer: An instance of the SentencePiece tokenizer used for tokenizing the text data
    Returns:
        A tuple containing the prepared training dataset, validation dataset, test datasets, and batch samplers
    """

    # 1. Create dataset objects for training, validation, and testing.
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    # Create the encoder
    spk_label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 1. IMPORTANT: Tell the encoder to handle unknown labels
    spk_label_encoder.add_unk('<unk>')

    # 2. Define Speaker ID label pipeline.
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        spk_id_prefix = spk_id.split("-")[0]
        yield spk_id_prefix
        # Add allow_unk=True here
        spk_id_encoded = spk_label_encoder.encode_sequence_torch([spk_id_prefix], allow_unk=True)
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # ONLY extract unique speaker prefixes from TRAINING set
    unique_spk_prefixes = set()

    df = pd.read_csv(hparams["train_csv"])
    if "spk_id" in df.columns:
        spk_ids = df["spk_id"].tolist()
        for spk_id in spk_ids:
            # LibriSpeech format is SPK-CHAPTER-UTT
            spk_id_prefix = spk_id.split("-")[0]
            unique_spk_prefixes.add(spk_id_prefix)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    
    # This will now only contain the training speakers (likely 251)
    spk_label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[unique_spk_prefixes],
    )

    # 3. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 4. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "char_list", "tokens_list", "tokens")
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 5. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens", "spk_id_encoded"],
    )

    # 6. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None

    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_val = hparams["dynamic_batch_sampler_val"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_val,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
            "verfification_trials": hparams["verification_trials"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
    )
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    sparse_brain = SparseBrain(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    for param in sparse_brain.modules.codec.parameters():
        param.requires_grad = False
    sparse_brain.modules.codec.eval() # Set to eval mode to stop dropout/batchnorm updates

    sparse_brain.tokenizer = tokenizer
    vocab_list = [
        tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())
    ]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"], vocab_list=vocab_list,
    )

    # adding objects to trainer:
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    # Measure time
    start_time = time.time()

    sparse_brain.fit(
        sparse_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    end_time = time.time()  # End the timer
    # Calculate elapsed time
    elapsed_time = (end_time - start_time) / 3600  # Convert to hours
    logger.info("Elapsed time %s hours", elapsed_time)

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        sparse_brain.hparams.output_wer_folder = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        sparse_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
    )

