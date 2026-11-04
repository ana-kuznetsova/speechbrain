# Speaker leakage into the content branch: problems and fixes

Summary of why speaker information leaked into the content code `H_cnt` in the
sparse disentanglement recipe, and what was changed to reduce it.

Files touched:

- [`speechbrain/lobes/models/sparse_disentangle.py`](../../../../speechbrain/lobes/models/sparse_disentangle.py)
- [`train_cnt.py`](./train_cnt.py)
- [`hparams/sparse_cnt_full.yaml`](./hparams/sparse_cnt_full.yaml)

The causes below come from reading the code and the training log. They are
hypotheses until confirmed with the `leak_EER` metric (see
[How to check it worked](#how-to-check-it-worked)).

## Problems

| # | Problem | Why it causes leakage |
|---|---------|-----------------------|
| 1 | No loss penalizes speaker information in `H_cnt`. | Reconstruction does not care which branch carries a component. CTC only needs content to be informative, and AAM only needs `h_spk` to be informative. The bidirectional LSTM sees the whole utterance and can read speaker identity from utterance-level statistics of `H_cnt`. |
| 2 | Content and speaker atoms are unconstrained. | Both live in one dictionary `W` and are only split by column index. Where their spans overlap, either branch can explain the same component of `z`. |
| 3 | `H_cnt` can represent anything `h_spk` can. | `H_cnt` is `(B, K, T)`, so a time-constant code is valid. The per-utterance mean of `z` (the speaker/channel offset) can be absorbed by the frame-wise content code. |
| 4 | Residual stacking pushes leftovers to content. | Layers 2-4 code the residual. Speaker information that layer 1's 32-d speaker code cannot hold is coded by later content atoms. |
| 5 | Codes were not sparse. | `l1_lambda` defaulted to `1e-4` (never set in the YAML or forwarded to the layers), far below the code scale. Logged `active_density_layer_*` was about 99.97%, so dense content codes had spare capacity for speaker information. Codes were also initialized with `randn` and run for only 5 steps, so they never converged and encoding was non-deterministic. |
| 6 | Loss weights unbalanced. | Reconstruction weight was 1.0 versus 2e6 (AAM), 8e5 (speaker reg) and 8000 (CTC). The dictionary was shaped mostly by classification gradients, not by a meaningful decomposition of `z`. |
| 7 | No reconstruction-only warmup. | `number_warmup_epochs` was commented out, so the dictionary was trained jointly from step 0 and the leakage was baked in early. |
| 8 | Stale speaker code in the reconstruction. | The old forward reconstructed with a speaker code from one iteration earlier than the returned `h_spk`. |

## Fixes

### Structural (`sparse_disentangle.py`)

All new options default to the old behaviour, so the other recipes that import
the module are unaffected.

| Option | Addresses | What it does |
|--------|-----------|--------------|
| `mean_split` | 3, 4 | Speaker branch explains only the masked per-utterance time-mean of `z`, solved on its own. Content branch explains the mean-removed signal, so it has no constant offset to absorb. Mean left over after one layer is handed to the next layer's speaker branch, not its content branch. Each block gets its own step size. |
| `coherence_penalty()` | 2 | Returns `cross` (squared Frobenius norm of `W_cnt^T W_spk`, zero when the subspaces are orthogonal) and `intra` (off-diagonal Gram energy inside each group). Summed over layers by `ResidualSparseDisentangle.coherence_penalty()`. |
| `relative_lambda` + `l1_lambda` | 5 | `l1_lambda` is now forwarded by `ResidualSparseDisentangle` and can be a fraction of `lambda_max` (per utterance), so it is independent of the scale of `z`. |
| `code_init: zeros / correlation` | 5 | Deterministic initialization instead of `randn` noise. |
| `use_fista` | 5 | Nesterov momentum, converging faster per unrolled step. |
| `speaker_first` | 3 | Gauss-Seidel order (speaker updated first, content sees the updated residual) for when `mean_split` is off. |
| Final reconstruction from final codes | 8 | Fixes the stale speaker code. This changes reconstruction in multi-layer stacks, so old checkpoints do not reproduce the same numbers. Layer 1 is bit-for-bit identical to the old code under default flags. |

### Losses and diagnostics (`train_cnt.py`)

| Item | Addresses | What it does |
|------|-----------|--------------|
| Coherence loss (`coherence_weight`, `intra_coherence_weight`) | 2 | Weighted coherence penalty, applied in warmup and joint training. |
| Dependence penalty (`dependence_weight`) | 1 | Mean squared cross-correlation between pooled content statistics (mean, std, mean-abs over time) and the speaker code. Linear dependence only. Skipped for batches smaller than 4. |
| Invariance loss (`invariance_weight`, `invariance_pitch_steps`) | 1 | Content code of a pitch/formant-shifted copy of the batch (random shift per batch) must match the original. Shifted branch runs without gradients, which keeps memory low. Joint TRAIN only. |
| `leak_EER` | diagnostic | Verification EER scored on the pooled content code in valid/test. About 50% means little leakage; a low value means speaker identity can be read from `H_cnt`. |
| Extra logging | diagnostic | Raw `ctc_loss`, `aam_loss`, `recon_loss`, regularizers, and the new terms are written to `train_log.txt` every `loss_log_interval` steps and averaged into the epoch lines. |

### Config (`hparams/sparse_cnt_full.yaml`)

- Warmup re-enabled: `number_warmup_epochs: 1` (fixes 7).
- Disentangle block: `num_steps: 8`, `l1_lambda: 0.2`, `relative_lambda: True`,
  `code_init: zeros`, `use_fista: True`, `mean_split: True`, `speaker_first: True`
  (the last is ignored while `mean_split` is on).
- New loss weights: `coherence_weight: 1.0e5`, `intra_coherence_weight: 2.0e4`,
  `dependence_weight: 5.0e5`, `invariance_weight: 5.0e5`.

## Not implemented

- Gradient reversal and any adversarial loss (deferred by request).
- CLUB mutual-information bound: needs a separately trained estimator.
- Rebalancing the existing weights (`spk_aam_loss_weight`, `ctc_weight`,
  `spk_reg_weight`, `content_reg_weight`, `sparse_loss_weight`): this needs your
  logged per-term magnitudes (problem 6 is only mitigated, not solved).

## Caveats

- The new weights and `l1_lambda: 0.2` are starting points, not tuned values.
  Compare the logged per-term values against the roughly 1e7 total loss. Tune
  `l1_lambda` using `active_density_layer_*` (lower lambda gives denser codes).
- Extra cost: `num_steps` 5 to 8, plus one pitch shift, codec encode and sparse
  coding pass per batch for the invariance loss. If you run out of memory, lower
  `num_steps` or set `invariance_weight: 0`.
- Start a fresh experiment rather than resuming old checkpoints.

## How to check it worked

1. `leak_EER` in valid lines should move toward 50% (higher is less leakage),
   while `EER` (from `h_spk`) stays low.
2. `active_density_layer_*` should drop well below 99%.
3. `dependence_loss`, `coherence_cross` and `invariance_loss` should decrease.
4. WER should not degrade much; if it does, lower `invariance_weight` or
   `dependence_weight`.

## Validation status

Tested on synthetic data in a throwaway environment: the module matches the old
code under default flags (single layer), `mean_split` removes the time-mean from
the content reconstruction, FISTA lowers reconstruction error, relative lambda
produces real sparsity, and the training code path runs in TRAIN, warmup and
VALID with finite losses and gradients. No full training run has been done yet.
