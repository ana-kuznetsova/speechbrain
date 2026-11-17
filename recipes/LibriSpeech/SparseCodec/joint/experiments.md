# Experiments: why ASR does not train, and how to balance leakage against ASR

## Why these experiments

In `/storage/anakuzne/exp/debug_auxiliary_loss_vals` (the current run) the CTC loss
stays at about 480-580 for all 20 epochs and valid WER is 99-140%. In the baseline run
`/storage/anakuzne/exp/debug` (no leakage fixes) CTC was already 155 at step 1600 and
reached about 30 by epoch 8 (WER about 38% at epoch 6). The ASR never started learning,
so the first job is to find which change blocks it. `leak_EER` (31-37%) is only
meaningful once ASR works, because an uninformative `h_cnt` also has a low leak.

Reference numbers to compare against (same epochs, valid set):

| run | CTC at step ~1600 | CTC at step ~3000 | WER epoch 2 | WER epoch 3 |
|-----|-------------------|-------------------|-------------|-------------|
| baseline `debug` | 155 | 112 | 56.8 | 49.7 |
| current `debug_auxiliary_loss_vals` | 579 | 528 | 104 | 99.0 |

A run "trains" if its CTC loss is clearly falling by the end of epoch 3 (baseline-like
trend), even if it is not as low as the baseline.

## Code changes that support the experiments

All new options default to the old behaviour. Set them on the command line with
`--name value`.

| Option / log field | Meaning |
|--------------------|---------|
| `penalty_start_epoch`, `penalty_ramp_epochs` | Dependence and invariance penalties are 0 before `penalty_start_epoch`, then ramp linearly to full strength over `penalty_ramp_epochs` epochs (0 = full strength at once). Epoch 1 is the reconstruction-only warmup. Example: start 5, ramp 3 gives scales 1/3, 2/3, 1 at epochs 5, 6, 7. The pitch-shift pass is skipped while the scale is 0. |
| `dependence_floor_scale` | Hinge for the dependence loss: only the part above `dependence_floor_scale / (batch_size - 1)` (the expected value for independent features) is penalised. 0 = plain penalty. |
| `invariance_margin` | Hinge for the invariance loss: only the part above the margin is penalised. 0 = plain penalty. The logged value in the current run is already about 0.05-0.1. |
| `l1_lambda`, `mean_split` | Now top-level YAML values (forwarded to the disentangler) so they can be set from the command line. Defaults unchanged (0.2, True). |
| `skip_test` | `True` exits after training, skipping the beam-search test. Use it for short runs. The test stage also deletes all but one checkpoint (it saves `WER: 0.0`). |
| `train rel_recon_err`, `valid rel_recon_err` | New. Masked `||z - z_hat||^2 / ||z||^2` of the disentangler reconstruction (0 = perfect, 1 = predicts nothing). The existing `recon_loss` is an unnormalised sum, flat at about 5e6 in both runs, so it cannot compare configurations. |
| `dependence_loss` | Now logged even when `dependence_weight` is 0, so leakage-related correlation can still be watched. |
| `penalty_scale` | Logged scale of the penalties at the current epoch. |
| `grad_log_interval` | Every N steps, writes a `train gradW_<term>: ...` line to `train_log.txt` with the gradient norm each weighted loss term (`recon`, `l12_content`, `l1_spk`, `coherence`, and in joint training also `ctc`, `aam`, `dependence`, `invariance`) puts on the dictionary `W`. 0 = off. Costs one extra backward pass per term at those steps. Used in Phase 3 to see whether the reconstruction loss is drowned out. |
| `invariance_num_shifts` | Number of different pitch shifts per batch (drawn without replacement from `invariance_pitch_steps`); the loss is averaged over them. Each one costs an extra encoder and sparse-coding pass. Default 1. |
| `invariance_target` | `content` (default): invariance on `h_cnt`, which trains the dictionary. `asr`: invariance on the ASR-encoder output of **detached** codes, so it trains only the ASR encoder and sends no gradient into the dictionary. LSTM encoder only. Dropout makes the two passes differ slightly, so the logged value has a noise floor. |

## Setup

Run from the `joint` directory. Replace the interpreter if you use a different env.

```bash
cd /home/anastasia/repos/speechbrain/recipes/LibriSpeech/SparseCodec/joint
conda activate /home/anastasia/envs/sb   # or your training env

DATA=/storage/shared/datasets/LibriSpeech/
EXP=/storage/anakuzne/exp
HP=hparams/sparse_cnt_full.yaml

# Ablation runs: 1 warmup epoch + 2 joint epochs, no test stage.
SHORT="--number_of_epochs 3 --skip_test True"
```

Each run writes to its own folder, so data prep and tokenizer training are repeated.
If you want to save that time, copy `train.csv`, `dev-clean.csv`, `veri_dev.txt` and
the tokenizer from an earlier run folder and add `--skip_prep True`.

Helper to summarise a finished (or running) run:

```bash
summ () {
  echo "== $1"
  grep "avg_over_last" $EXP/$1/train_log.txt | awk 'NR%5==0' \
    | grep -oE "steps: [0-9]+|train (ctc_loss|rel_recon_err|dependence_loss|invariance_loss): [^,]+" | paste -sd' ' | fold -w 200
  grep "lr:" $EXP/$1/train_log.txt \
    | grep -oE "^epoch: [0-9]+|valid (WER|EER|leak_EER|rel_recon_err): [^,]+" | paste -sd' ' | sed 's/epoch:/\nepoch:/g'
}
```

For comparison with the baseline CTC values, look at the `ctc_loss` at steps around
1600-3000 (the first joint epoch ends at about step 1476 + 1476 = 2952 per epoch).

## Phase 1: which change blocks ASR (3 epochs each)

Run in this order. Stop early if a run's CTC is still flat at about 500 after the first
joint epoch.

### A. No leakage penalties, sparse settings unchanged

Isolates the sparse coding itself (`l1_lambda: 0.2`, `mean_split`, FISTA, relative lambda).

```bash
python train_cnt.py $HP $SHORT --data_folder $DATA --output_folder $EXP/abl_A_nopen \
  --dependence_weight 0 --invariance_weight 0 --coherence_weight 0 --intra_coherence_weight 0
```

- CTC still flat: the cause is in the sparse coding setup. Skip B and C, go to D and E.
- CTC falls like the baseline: the sparse coding is fine and the penalties block ASR. Go to B and C.

### B. Invariance loss off only

```bash
python train_cnt.py $HP $SHORT --data_folder $DATA --output_folder $EXP/abl_B_noinv \
  --invariance_weight 0
```

### C. Dependence loss off only

```bash
python train_cnt.py $HP $SHORT --data_folder $DATA --output_folder $EXP/abl_C_nodep \
  --dependence_weight 0
```

Reading B and C: whichever one makes CTC fall is the blocker. If both are needed to
get CTC down (A works, B and C alone do not), the penalties together are too strong;
move to Phase 2.

### D. Less sparse content code

```bash
python train_cnt.py $HP $SHORT --data_folder $DATA --output_folder $EXP/abl_D_l1_0.05 \
  --l1_lambda 0.05
```

Check `valid rel_recon_err` and `active_density_layer_*` as well as CTC.

### E. Without the mean split

```bash
python train_cnt.py $HP $SHORT --data_folder $DATA --output_folder $EXP/abl_E_nomeansplit \
  --mean_split False
```

### Record the results

| run | CTC step ~3000 | valid WER ep 3 | valid rel_recon_err | valid EER | valid leak_EER | notes |
|-----|----------------|----------------|---------------------|-----------|----------------|-------|
| A no penalties | 3.15e+02 | 72.77 | 9.58e-01| 4.54e-02| 2.09e-01| |
| B no invariance | 2.15e+02| 68.54| 8.92e-01| 4.94e-02| 2.22e-01| |
| C no dependence | 5.11e+02 | 1.11e+02| 9.97e-0 |4.90e-02 | 3.10e-01 | |
| D l1_lambda 0.05 | 2.69e+02| 74.14| 9.96e-01| 5.06e-02| 2.33e-01| |
| E no mean_split | | | | | | |

## Phase 2: schedule and hinge (longer runs)

Use the best configuration from Phase 1 as the base (put its overrides in `BASE`).
For example, if the penalties were the blocker, `BASE=""`. If `l1_lambda` was the blocker,
`BASE="--l1_lambda 0.05"`. These runs use 12 epochs and do the test stage.

```bash
BASE=""
LONG="--number_of_epochs 12"
```

### F. Penalty ramp (delay, then ramp)

Penalties are 0 in epochs 2-4 so CTC can learn, then ramp 1/3, 2/3, 1 in epochs 5-7.

```bash
python train_cnt.py $HP $LONG $BASE --data_folder $DATA --output_folder $EXP/exp_F_ramp \
  --penalty_start_epoch 5 --penalty_ramp_epochs 3
```

Watch whether CTC and WER keep improving once `penalty_scale` reaches 1, and where
`leak_EER` ends up.

### G. Ramp plus hinge

Only penalise dependence above the noise floor and invariance above a margin, so the
penalties stop pushing once they are about as small as they can get.

```bash
python train_cnt.py $HP $LONG $BASE --data_folder $DATA --output_folder $EXP/exp_G_ramp_hinge \
  --penalty_start_epoch 5 --penalty_ramp_epochs 3 \
  --dependence_floor_scale 1.0 --invariance_margin 0.05
```

### H. Weaker penalties (weight sweep, if F or G still hurt WER)

```bash
for w in 0.5 0.2 0.1; do
  python train_cnt.py $HP $LONG $BASE --data_folder $DATA --output_folder $EXP/exp_H_w$w \
    --penalty_start_epoch 5 --penalty_ramp_epochs 3 \
    --dependence_weight $(python -c "print(5e7*$w)") \
    --invariance_weight $(python -c "print(2.5e7*$w)")
done
```

### Record the results

| run | WER (best epoch) | valid rel_recon_err | valid EER | valid leak_EER | notes |
|-----|------------------|---------------------|-----------|----------------|-------|
| F ramp | | | | | |
| G ramp + hinge | | | | | |
| H w=0.5 | | | | | |
| H w=0.2 | | | | | |
| H w=0.1 | | | | | |

Pick the run with the best trade-off between WER, `rel_recon_err` and `leak_EER`.
Confirm with the probe notebook (`sparse_inference.ipynb`) after a run that also did
the test stage; it reads the one checkpoint that is kept.

## Phase 3: protect the reconstruction

### Why

In the Phase 1 runs `rel_recon_err` is about 1.0 at the end of the warmup epoch, and with the
invariance loss and `mean_split` together (C, D) it stays at 0.995 for the whole run:

| run | `mean_split` | invariance | final valid `rel_recon_err` | WER (last epoch) |
|-----|--------------|------------|-----------------------------|------------------|
| B | on | off | 0.81 | 46.7 |
| D | on | on | 0.996 | 43.4 |
| E | off | on | 0.70 | 51.7 |

A gradient-norm check on the D, E and B checkpoints (one batch of 8 dev utterances, one +3 semitone
shift, gradient on the dictionary `W` only) gave:

| weighted term | D | E | B |
|---------------|---|---|---|
| reconstruction | 1.0e4 | 3.7e4 | 2.4e4 |
| L1 on speaker code | 1.4e6 | 1.0e6 | 1.1e6 |
| invariance | 7.5e6 | 3.2e5 | - |
| dependence | 3.2e7 | 1.3e6 | 2.2e6 |

So the reconstruction loss has 100-1000x less influence on the dictionary than the other terms, in every
run. Phase 3 tests three fixes: (1) rebalance the weights, (2) learn the reconstruction in a longer
warmup before the penalties start, (3) reduce how strongly the invariance loss pulls on the dictionary.
Freezing the dictionary is not tried here.

### Setup for Phase 3

```bash
WARM="--number_of_epochs 2 --number_warmup_epochs 2 --skip_test True --grad_log_interval 100"
JOINT="--number_of_epochs 10 --number_warmup_epochs 3 --skip_test True --grad_log_interval 200"

# 100x lower regularisers, coherence and penalties, so the reconstruction loss is not drowned out.
# The regulariser and coherence values are the W2 setting (rel_recon_err 0.155 after 2 warmup epochs,
# against 0.132 for reconstruction only); the dependence weight is a starting value.
# Adjust them with the gradW_ log lines.
REBAL="--spk_reg_weight 8.0e3 --content_reg_weight 5.0 --coherence_weight 1.0e5 --intra_coherence_weight 2.0e4 --dependence_weight 5.0e5"
# Invariance weight (100x lower than before); each joint run below passes it once.
# Do not repeat an option on one command line: duplicate overrides are rejected with a YAML error.
INV="--invariance_weight 2.5e5"
```

Helper to read the gradient lines of a run (the lines written by `grad_log_interval`):

```bash
gsumm () {
  echo "== $1"
  grep "gradW_" $EXP/$1/train_log.txt | awk 'NR%3==1' | while read -r line; do
    echo "$line" | grep -oE "^epoch: [0-9]+|steps: [0-9]+|train gradW_[a-z0-9_]+: [^,]+" | paste -sd' '
  done
}
```

Read it as the share of each term relative to `gradW_recon`. The goal is a reconstruction gradient
within about 10x of the largest other term, and no term 1000x above it.

### Fixes 1 and 2: warmup runs (2 warmup epochs, a few minutes each)

These runs have no ASR and no penalties, only reconstruction, the L1 terms and the coherence term.
Compare the final valid `rel_recon_err` (it is 1.00 after one warmup epoch in all earlier runs) and the
`gradW_` lines.

#### W0. Control (current weights)

```bash
python train_cnt.py $HP $WARM --data_folder $DATA --output_folder $EXP/rec_W0_control
```

#### W1. Lower L1 regularisers (fix 1)

```bash
python train_cnt.py $HP $WARM --data_folder $DATA --output_folder $EXP/rec_W1_lowreg \
  --spk_reg_weight 8.0e3 --content_reg_weight 5.0
```

#### W2. Lower L1 regularisers and coherence (fix 1)

On a fresh dictionary the coherence term can have a very large gradient (1e8 in a synthetic test, against 1e2 for
the reconstruction); it is about 1e-4 later in training, so this mostly matters at the start of the warmup.

```bash
python train_cnt.py $HP $WARM --data_folder $DATA --output_folder $EXP/rec_W2_lowreg_lowcoh \
  --spk_reg_weight 8.0e3 --content_reg_weight 5.0 --coherence_weight 1.0e5 --intra_coherence_weight 2.0e4
```

#### W3. Higher reconstruction weight instead (fix 1)

```bash
python train_cnt.py $HP $WARM --data_folder $DATA --output_folder $EXP/rec_W3_recon100 \
  --sparse_loss_weight 100
```

#### W4. Reconstruction only (reference)

Shows how low `rel_recon_err` can go with this dictionary size and `l1_lambda`. If this is also near 1, the problem is
not the weights and the sparse coder itself (`l1_lambda`, dictionary size, `mean_split`) has to change.

```bash
python train_cnt.py $HP $WARM --data_folder $DATA --output_folder $EXP/rec_W4_reconly \
  --spk_reg_weight 0 --content_reg_weight 0 --coherence_weight 0 --intra_coherence_weight 0
```

| run | valid `rel_recon_err` ep 2 | gradW_recon | gradW_l1_spk | gradW_coherence | notes |
|-----|----------------------------|-------------|--------------|-----------------|-------|
| W0 control | 1.00e+00 | 2.88e+03|1.48e+06| 1.57e+04| |
| W1 low reg | 1.64e-01|3.59e+04|7.11e+03|3.55e+04| |
| W2 low reg + coherence | 1.55e-01 | 3.30e+04|7.32e+03| 1.31e+04| |
| W3 recon x100 |1.55e-01| 3.31e+06| 7.33e+05 | 1.32e+06 | |
| W4 recon only |1.32e-01 | 1.88e+04| 0|0 | |

W2 and W3 reach the same `rel_recon_err`, and their gradients are in the same proportion (recon : l1_spk : coherence
= 4.5 : 1 : 1.8); W3 is W2 scaled by 100. W2 was chosen because it keeps the reconstruction gradient in the same range as the
CTC, AAM and penalty terms, whereas W3 would make it 100x stronger than the untouched CTC and AAM terms. `REBAL` below
uses the W2 flags. After the J runs check that `coherence_cross` stays small (about 1e-4 in earlier runs), since the coherence
weight is now lower.

### Fixes 1, 2 and 3: joint runs (10 epochs, 3 warmup epochs)

All joint runs keep the default sparse settings (`mean_split` on, `l1_lambda` 0.2) and `dependence_weight`
unless stated. Compare against B (`rel_recon_err` 0.81, WER 46.7 after 11 epochs) and D (0.996, WER 43.4).
A good run has `rel_recon_err` falling instead of staying flat near 1, WER within a few points of B, and
`leak_EER` not worse than about 0.2.

#### J1. Rebalanced weights, penalties on from the first joint epoch (fix 1)

```bash
python train_cnt.py $HP $JOINT $REBAL $INV --data_folder $DATA --output_folder $EXP/rec_J1_rebal
```

#### J2. Rebalanced weights, longer warmup, delayed and ramped penalties (fixes 1 and 2)

Epochs 1-3 are reconstruction only. Epochs 4-5 train the ASR and speaker heads with the penalties off. The penalties
start at epoch 6 (scale 0.5) and are at full strength from epoch 7 (`penalty_start_epoch` 6, ramp 2). `JBASE` is
reused by J3-J6 (define it again if you open a new shell).

```bash
JBASE="$JOINT $REBAL --penalty_start_epoch 6 --penalty_ramp_epochs 2"
python train_cnt.py $HP $JBASE $INV --data_folder $DATA --output_folder $EXP/rec_J2_rebal_delay
```

#### J3. Weaker invariance pull on the dictionary (fix 3)

Invariance weight another 10x lower, with a margin so it stops pushing once the loss is small.
The logged invariance value is about 0.03-0.08 in earlier runs; the margin of 0.05 is a starting point.

```bash
python train_cnt.py $HP $JBASE --data_folder $DATA --output_folder $EXP/rec_J3_weakinv \
  --invariance_weight 2.5e4 --invariance_margin 0.05
```

#### J4. Several pitch shifts per batch (fix 3)

Less dependence of the gradient on one random shift. Costs two extra encoder and sparse-coding passes per batch.

```bash
python train_cnt.py $HP $JBASE --data_folder $DATA --output_folder $EXP/rec_J4_3shifts \
  $INV --invariance_num_shifts 3
```

#### J5. Invariance on the ASR features (fix 3)

The invariance loss trains only the ASR encoder and sends no gradient into the dictionary, so it cannot hurt the
reconstruction. Since it can no longer harm the reconstruction its weight is 10x higher than in J2.

```bash
python train_cnt.py $HP $JBASE --data_folder $DATA --output_folder $EXP/rec_J5_asrinv \
  --invariance_target asr --invariance_weight 2.5e6
```

Check `gradW_invariance`: it must be 0 for this run. Note that this removes the direct pressure on the content code, so
`leak_EER` may not improve as much; check it before drawing conclusions.

#### J6. Combination (only after J3-J5)

Combine what helped, for example the ASR-feature invariance with 3 shifts:

```bash
python train_cnt.py $HP $JBASE --data_folder $DATA --output_folder $EXP/rec_J6_combo \
  --invariance_target asr --invariance_weight 2.5e6 --invariance_num_shifts 3
```

| run | valid `rel_recon_err` (last) | train CTC (last) | WER (best epoch) | valid EER | valid leak_EER | notes |
|-----|------------------------------|------------------|------------------|-----------|----------------|-------|
| J1 rebalanced | | | | | | |
| J2 + delay and ramp | 1.77e-01| 38.98|42.52 |1.19e-01| 1.81e-01| |
| J3 weak invariance |8.26e-01 | 46.70| 46.14| 4.10e-02|2.01e-01 | |
| J4 3 shifts |8.57e-01 | 43.31 | 44.46|4.18e-02 | 1.70e-01| |
| J5 ASR-feature invariance | | | | | | |
| J6 combination | | | | | | |

After a good run, also repeat the test stage (drop `--skip_test True`) and run section 7 of `sparse_inference.ipynb`
to listen to the reconstruction; the metric to watch there is the log-mel L1 of the sparse row against the DAC rows.

## Phase 4: per-layer weighting (not implemented yet)

Layer 1 carries most of the leakage (52% speaker ID from layer 1 alone against 16-19%
for the other layers). Once Phase 1-2 gives a run where ASR trains, apply the dependence
and invariance penalties per layer with a larger weight on layer 1. This needs the model
to return the per-layer content codes, so it is left until the earlier results show it is
worth the change.
