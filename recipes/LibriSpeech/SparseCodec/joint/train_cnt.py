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
import torch.nn.functional as F

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

def set_sparse_only_training(residual_sparse_module, enable_sparse_only):
    """
    If enable_sparse_only is True, freeze all parameters except the sparse dictionary modules
    and the decoder adapter needed to project sparse representations back to the input space.
    If False, unfreeze all parameters in ResidualSparseDisentangle.
    """
    for name, param in residual_sparse_module.named_parameters():
        if enable_sparse_only:
            should_train = (
                "sparse_module_list" in name or "decoder_adapter" in name
            )
            param.requires_grad = should_train
            logger.info(
                "%s parameter: %s",
                "Enabling" if should_train else "Freezing",
                name,
            )
        else:
            param.requires_grad = True


# Define training procedure
class SparseBrain(sb.core.Brain):
    def compute_forward(self, batch: Dict[str, torch.Tensor], stage: sb.Stage) -> Tuple:
        """Forward computations from waveform batches to output probabilities and embeddings.
        
        Args:
            batch: SpeechBrain PaddedBatch containing input waveforms and targets.
            stage: Current training stage (TRAIN, VALID, TEST).
            
        Returns:
            Tuple containing CTC probabilities, predicted hypotheses, lengths, 
            speaker logits, reconstruction metrics, and latent representations:
                - p_ctc: Log-probabilities from the CTC head
                - pred_hyps: Predicted hypotheses (decoded sequences)
                - wav_lens: Lengths of the input waveforms
                - h_spk: Speaker latent representations
                - h_recon: Reconstructed latent representations from the sparse module
                - recon_loss: Reconstruction loss (MSE) for the latent representations
                - l1_reg_content: L1 regularization term for content features
                - l1_reg_speaker: L1 regularization term for speaker features
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # 1. Extract raw features from DAC codec encoder: [B, Channels, T]
        enc_out = self.modules.codec.encoder(wavs.unsqueeze(1))

        # 2. Residual Sparse Disentanglement Pass
        (   recon_loss, 
            l12_reg_content, 
            l12_reg_speaker, 
            h_cnt,
            h_spk,
            h_spk_mat,
            h_all,
        ) = self.modules.disentangle(
            enc_out,
            stage=stage,
            wav_lens=wav_lens
        )

        # 3. ASR Encoder Forward Pass
        # Permute content features to [B, T, D_cnt] for sequence modeling
        h_cnt_seq = h_cnt.permute(0, 2, 1)

        if isinstance(self.modules.asr_encoder, sb.nnet.RNN.LSTM):
            enc_out_asr, _ = self.modules.asr_encoder(h_cnt_seq, lengths=wav_lens)
        else:
            # Conformer / Transformer models requiring target tokens or mask
            target_tokens, _ = batch.tokens
            enc_out_asr, _, _, _ = self.modules.asr_encoder(
                h_cnt_seq, target_tokens, wav_lens
            )

        logits = self.modules.ctc_lin(enc_out_asr)
        p_ctc = self.hparams.log_softmax(logits)

        # 4. Decode ASR Predictions for Evaluation
        pred_hyps = None
        if stage == sb.Stage.VALID:
            pred_hyps = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            # Fixed: reference searcher via hparams
            pred_hyps = test_searcher(p_ctc, wav_lens)

        # 5. Speaker Classification Head Forward Pass
        # h_spk shape is (B, K_spk_total, 1) -> Squeeze to (B, K_spk_total)
        h_spk_vec = h_spk.squeeze(-1)
        spk_logits = self.modules.spk_classifier(h_spk_vec)

        # 6. Collect Utterance Embeddings for Speaker Verification Scoring
        if stage != sb.Stage.TRAIN:
            if not hasattr(self, "eval_spk_embs"):
                self.eval_spk_embs = {}
            utt_ids = batch.id
            for utt_id, emb in zip(utt_ids, h_spk_vec):
                self.eval_spk_embs[utt_id] = emb.cpu().detach()
        #logging.info("Debug losses: recon_loss=%f, l12_reg_content=%f, l12_reg_speaker=%f", recon_loss.item(), l12_reg_content.item(), l12_reg_speaker.item())
        return (
            p_ctc,
            pred_hyps,
            wav_lens,
            spk_logits,
            recon_loss,
            h_cnt,
            h_spk,
            h_spk_mat,
            h_all,
            l12_reg_content,
            l12_reg_speaker,
        )

    def compute_objectives(
        self, predictions: tuple, batch: dict, stage: sb.Stage
    ) -> torch.Tensor:
        """Computes combined ASR, reconstruction, and speaker losses.

        Args:
            predictions: Tuple containing the outputs from compute_forward().
            batch: PaddedBatch containing waveforms and target tokens/labels.
            stage: Current training stage (TRAIN, VALID, TEST).

        Returns:
            Combined scalar loss tensor for backpropagation.
        """
        # 1. Unpack 9 prediction outputs matching updated compute_forward
        (
            p_ctc,
            pred_hyps,
            wav_lens,
            spk_logits,
            recon_loss,
            h_cnt,
            h_spk,
            h_spk_mat,
            h_all,
            l12_reg_content,
            l12_reg_speaker,
        ) = predictions

        uttid = batch.id

        # ------------------------------------------------------------------
        # PATH A: Sparse-Only Warmup Training Phase
        # ------------------------------------------------------------------
        if stage == sb.Stage.TRAIN and getattr(self, "sparse_only", False):
            total_loss = (
                recon_loss * getattr(self.hparams, "sparse_loss_weight", 1.0)
                + l12_reg_content * getattr(self.hparams, "content_reg_weight", 1e-4)
                + l12_reg_speaker * getattr(self.hparams, "spk_reg_weight", 1e-4)
            )
            # Store for epoch-level summary stats tracking
            if not hasattr(self, "stage_stats"):
                self.stage_stats = {}
            return total_loss

        # ------------------------------------------------------------------
        # PATH B: Standard Multi-Task Joint Training Phase
        # ------------------------------------------------------------------
        tokens, tokens_lens = batch.tokens
        spk_targets, _ = batch.spk_id_encoded

        # 2. Compute CTC loss for ASR
        ctc_batch_loss = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens, reduction=self.hparams.loss_reduction
        )
        ctc_batch_loss = ctc_batch_loss * self.hparams.ctc_weight

        # 3. Compute Regularization & Disentanglement Losses
        spk_reg_loss = l12_reg_speaker * self.hparams.spk_reg_weight
        content_reg_loss = l12_reg_content * self.hparams.content_reg_weight

        total_loss = ctc_batch_loss + spk_reg_loss + content_reg_loss

        # 4. Add Reconstruction & Speaker Classification Losses (TRAIN & VALID)
        if stage in (sb.Stage.TRAIN, sb.Stage.VALID):
            recon_batch_loss = recon_loss * getattr(
                self.hparams, "sparse_loss_weight", 1.0
            )
            
            # AAM / Angular Softmax Loss
            batch_aam_loss = self.hparams.spk_aam_loss(spk_logits, spk_targets)
            batch_aam_loss = batch_aam_loss * getattr(
                self.hparams, "spk_aam_loss_weight", 1.0
            )

            total_loss = total_loss + recon_batch_loss + batch_aam_loss

        # 5. Compute Sparsity & Activity Metrics for Monitoring
        # Inside compute_objectives (for TRAIN and VALID):
        if stage in (sb.Stage.TRAIN, sb.Stage.VALID):
            with torch.no_grad():
                h_abs = h_all.abs()
                n_elements = h_abs.numel()
                n_zero = (h_abs < 1e-4).sum().float()
                sparsity_level = (n_zero / n_elements) * 100.0
                #logging.info(f"Sparsity Level: {sparsity_level.item():.2f}%")

                # h_all: [B, 2, L*K, T] -> [B, 2, L, K, T]; density per layer
                n_layers = self.hparams.num_sparse_layers
                B_, _, _, T_ = h_abs.shape
                active_density_per_layer = (
                    (h_abs > 1e-4)
                    .float()
                    .view(B_, 2, n_layers, -1, T_)
                    .mean(dim=(0, 1, 3, 4))
                    * 100.0
                )

                # Ensure dictionary exists on self
                if not hasattr(self, "stage_stats") or self.stage_stats is None:
                    self.stage_stats = {}

                # Safely append per-batch values
                self.stage_stats.setdefault("sparsity", []).append(sparsity_level.item())
                for i, density in enumerate(active_density_per_layer):
                    self.stage_stats.setdefault(f"active_density_layer_{i}", []).append(density.item())
                
        # 6. Evaluation Metrics Accumulation (VALID & TEST)
        if stage != sb.Stage.TRAIN and not getattr(self, "sparse_only", False):
            # Decode token hypotheses to text words
            if stage == sb.Stage.VALID:
                predicted_words = self.tokenizer(pred_hyps, task="decode_from_list")
            elif stage == sb.Stage.TEST:
                predicted_words = [hyp[0].text.split(" ") for hyp in pred_hyps]

            # Accumulate WER & Speaker Accuracy Metrics
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(uttid, predicted_words, target_words)

            spk_predictions = torch.argmax(spk_logits, dim=1)
            self.spk_error_metrics.append(uttid, spk_predictions, spk_targets)

        return total_loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()
        # NOTE: (anakuzne) change max_key to None for quantizer only inference
        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
            max_num_checkpoints=self.hparams.avg_checkpoints,
        )

        ckpt = sb.utils.checkpoints.average_checkpoints(ckpts, recoverable_name="model")

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()



    def on_stage_start(self, stage: sb.Stage, epoch: int = None):
        """Gets called at the beginning of each epoch / stage.

        Initializes loss accumulators, evaluation metrics, embedding dictionaries,
        and handles parameter freeze/unfreeze state transitions for warmup.
        """
        # 1. Reset loss accumulators across ALL stages (TRAIN, VALID, TEST)
        self.recon_loss_total = 0.0
        self.recon_loss_count = 0

        # 2. Reset speaker evaluation embedding store during non-training stages
        if stage != sb.Stage.TRAIN:
            self.eval_spk_embs = {}

        # 3. Determine if current epoch is in the sparse-only warmup phase
        warmup_epochs = getattr(self.hparams, "number_warmup_epochs", 0)
        is_warmup = (
            epoch is not None and warmup_epochs > 0 and epoch <= warmup_epochs
        )
        self.sparse_only = is_warmup

        # 4. Initialize evaluation metrics when downstream tasks (ASR / verification) are active
        if stage != sb.Stage.TRAIN and not (
            stage == sb.Stage.VALID and self.sparse_only
        ):
            self.wer_metric = self.hparams.error_rate_computer()
            self.spk_error_metrics = self.hparams.spk_error_stats()

        # 5. Apply Parameter Freezing / Unfreezing logic
        if stage == sb.Stage.TRAIN:
            if self.sparse_only:
                # Rank-0 guarded logging to prevent multi-GPU output duplication
                if not hasattr(self, "is_main_process") or self.is_main_process:
                    logger.info(
                        "Epoch %d: Sparse-only warmup active. "
                        "Only SparseDisentangle parameters will receive gradients.",
                        epoch,
                    )
                set_sparse_only_training(
                    self.modules.disentangle, enable_sparse_only=True
                )
            else:
                set_sparse_only_training(
                    self.modules.disentangle, enable_sparse_only=False
                )
    


    def on_stage_end(self, stage: sb.Stage, stage_loss: float, epoch: int = None):
        """Called at the end of an epoch to aggregate statistics, compute metrics,
        update learning rates, and save model checkpoints.
        """
        stage_stats = {"loss": stage_loss}

        # 1. Aggregate Sparsity and Layer Activity Metrics collected during batch forward passes
        if hasattr(self, "stage_stats") and isinstance(self.stage_stats, dict):
            for key, values in self.stage_stats.items():
                if isinstance(values, list) and len(values) > 0:
                    stage_stats[key] = sum(values) / len(values)
            # Clear accumulators for the next stage
            self.stage_stats = {}

        # 2. Stage-Specific Statistics Setup
        if stage == sb.Stage.TRAIN:
            # Cache full training stats (including sparsity/densities) for valid_stats output call
            self.train_stats = stage_stats

        # 3. Validation and Testing Metrics (WER, SER, Cosine EER)
        elif not (stage == sb.Stage.VALID and getattr(self, "sparse_only", False)):
            current_epoch = (
                self.hparams.epoch_counter.current if epoch is None else epoch
            )
            valid_search_interval = getattr(
                self.hparams, "valid_search_interval", 1
            )

            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["ErrorRate"] = self.spk_error_metrics.summarize(
                    "average"
                )

                # --- Speaker Verification Cosine EER Calculation ---
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
                        continue

                    spk_emb1 = spk_emb1.reshape(-1)
                    spk_emb2 = spk_emb2.reshape(-1)

                    if spk_emb1.numel() != spk_emb2.numel():
                        logger.warning(
                            "Mismatched embedding sizes for %s (%d) and %s (%d). Skipping.",
                            utt1,
                            spk_emb1.numel(),
                            utt2,
                            spk_emb2.numel(),
                        )
                        continue

                    cos_sim = similarity(spk_emb1, spk_emb2)
                    if str(label).strip() == "1":
                        positive_scores.append(cos_sim.item())
                    else:
                        negative_scores.append(cos_sim.item())

                # Guard against empty verification scores
                if len(positive_scores) > 0 and len(negative_scores) > 0:
                    pos_tensor = torch.tensor(positive_scores)
                    neg_tensor = torch.tensor(negative_scores)
                    eer, _ = self.hparams.spk_verification_metrics(
                        pos_tensor, neg_tensor
                    )
                    stage_stats["EER"] = eer
                else:
                    logger.warning(
                        "Insufficient evaluation trial pairs found to compute EER."
                    )
                    stage_stats["EER"] = float("nan")

                if not hasattr(self, "is_main_process") or self.is_main_process:
                    logger.info(
                        "Statistics for epoch %s: EER %s, WER %s",
                        str(epoch),
                        str(stage_stats.get("EER", "N/A")),
                        str(stage_stats.get("WER", "N/A")),
                    )

        # 4. End-of-Validation Operations
        if stage == sb.Stage.VALID:
            if getattr(self, "sparse_only", False):
                if not hasattr(self, "is_main_process") or self.is_main_process:
                    logger.info(
                        "Skipping WER, speaker error, and EER during sparse-only warmup."
                    )

            # Scheduler step / LR retrieval
            scheduler_name = type(self.hparams.scheduler).__name__
            if scheduler_name == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif scheduler_name == "LinearNoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                # Safe generic fallback for PyTorch / custom schedulers
                lr = self.optimizer.param_groups[0]["lr"]

            steps = getattr(self, "optimizer_step", 0)
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            # Log training vs validation stats (both now contain their aggregated sparsity metrics)
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=getattr(self, "train_stats", {}),
                valid_stats=stage_stats,
            )

            # Checkpoint saving logic
            if getattr(self, "sparse_only", False):
                self.checkpointer.save_checkpoint(meta={"epoch": epoch})
            else:
                self.checkpointer.save_and_keep_only(
                    meta={
                        "WER": stage_stats.get("WER", stage_loss),
                        "epoch": epoch,
                    },
                    min_keys=["WER"],
                    num_to_keep=getattr(self.hparams, "avg_checkpoints", 1),
                )

        # 5. End-of-Test Operations
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            is_main = getattr(self, "is_main_process", True)
            if is_main and hasattr(self.hparams, "output_wer_folder"):
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
    spk_label_encoder.add_unk("<unk>")

    # 2. Define Speaker ID label pipeline.
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        spk_id_prefix = spk_id.split("-")[0]
        yield spk_id_prefix
        # Add allow_unk=True here
        spk_id_encoded = spk_label_encoder.encode_sequence_torch(
            [spk_id_prefix], allow_unk=True
        )
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
    
    if hasattr(hparams, "pretrainer") and hparams["pretrainer"] is not None:
        # Load collected checkpoints
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()
        logger.info("Pretrained model loaded from %s", hparams["pretrainer"])

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
    sparse_brain.modules.codec.eval()  # Set to eval mode to stop dropout/batchnorm updates

    sparse_brain.tokenizer = tokenizer
    vocab_list = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"],
        vocab_list=vocab_list,
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
