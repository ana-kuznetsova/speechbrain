#!/usr/bin/env python3
"""Recipe for training a sparse codec-based ASR system on LibriSpeech
in a discriminative style. The system is using ASR downstream with CTC+NLL loss 
and a Transformer-based architecture. And a soekaer recongitnion ECAPA head with AAM loss.


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

import librosa
import pandas as pd
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import if_main_process, run_on_main

logger = logging.getLogger(__name__)


# Define training procedure
class SparseBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)  # [B, T]

        # Codec forward pass
        (
            in_toks,
            _,
            _,
            commitment_loss,
            codebook_loss,
            sparse_loss,
            l1_reg_spk,
            l1_reg_cont,
        ) = self.hparams.codec(
            wavs, n_quantizers=self.hparams.num_codebooks
        )

        # ASR head forward pass
        in_toks = in_toks.transpose(1, 2)
        enc_out, _ = self.modules.asr_encoder(in_toks, lengths=wav_lens)
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        pred_hyps = None
        if stage == sb.Stage.VALID:
            pred_hyps = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            pred_hyps = test_searcher(p_ctc, wav_lens)

        # Spekaer classification head forward pass
        spk_emb = self.modules.spk_encoder(in_toks)
        spk_logits = self.modules.spk_classifier(spk_emb).squeeze(1)

        return (
            p_ctc, # ctc probabilities
            pred_hyps, # predicted hypotheses (token ids)
            wav_lens,
            spk_logits,
            commitment_loss,
            codebook_loss,
            sparse_loss,
            l1_reg_spk,
            l1_reg_cont,
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes combined ASR + L1 per attrribute regularizaer."""

        (
            p_seq,
            pred_hyps,
            wav_lens,
            spk_logits,
            _,
            _,
            sparse_loss,
            l1_reg_spk,
            l1_reg_cont,
        ) = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens
        spk_targets, _ = batch.spk_id_encoded

        # Convert spk_targets to one-hot
        spk_targets = torch.nn.functional.one_hot(
            spk_targets, num_classes=self.hparams.out_n_neurons
        ).squeeze(1)

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        ctc_batch_loss = self.hparams.ctc_cost(
            p_seq, tokens, wav_lens, tokens_lens, reduction=self.hparams.loss_reduction
        )
        ctc_batch_loss = ctc_batch_loss * self.hparams.ctc_weight
        sparse_batch_loss = sparse_loss * self.hparams.sparse_loss_weight

        batch_aam_loss = self.hparams.spk_aam_loss(spk_logits, spk_targets.transpose(0, 1))
        batch_aam_loss = batch_aam_loss * self.hparams.spk_aam_loss_weight

        spk_reg_loss = l1_reg_spk * self.hparams.spk_reg_weight
        content_reg_loss = l1_reg_cont * self.hparams.content_reg_weight

        loss = ctc_batch_loss + sparse_batch_loss + batch_aam_loss + spk_reg_loss + content_reg_loss

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = self.tokenizer(
                pred_hyps, task="decode_from_list"
            )
        elif stage == sb.Stage.TEST:
            predicted_words = [
                hyp[0].text.split(" ") for hyp in pred_hyps
            ]

        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
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
        print("DEBUG ckpts:", ckpts, max_key, min_key)
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError

            optimizer = self.optimizer.__class__.__name__
            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
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
                with open(
                    self.hparams.output_wer_folder, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if (
            should_step
            and type(self.hparams.scheduler).__name__ == "LinearNoamScheduler"
        ):
            self.hparams.scheduler(self.optimizer)


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    spk_label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 1. Define Speaker ID label pipeline.
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        spk_id_prefix = spk_id.split("-")[0]
        yield spk_id_prefix
        spk_id_encoded = spk_label_encoder.encode_sequence_torch([spk_id_prefix])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Extract unique speaker prefixes from training set
    unique_spk_prefixes = set()
    spk_ids = pd.read_csv(hparams["train_csv"])["spk_id"].tolist()
    for spk_id in spk_ids:
        spk_id_prefix = spk_id.split("-")[0]
        unique_spk_prefixes.add(spk_id_prefix)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    spk_label_encoder.load_or_create(
        path=lab_enc_file, from_iterables=[unique_spk_prefixes],
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sample_rate = librosa.get_samplerate(wav)
        resampled = torchaudio.transforms.Resample(
            sample_rate, hparams["sample_rate"],
        )(sig)
        # resampled = resampled.unsqueeze(0)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens", "spk_id_encoded"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
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
