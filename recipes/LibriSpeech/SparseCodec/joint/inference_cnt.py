#!/usr/bin/env python3
"""
Script for running inference using a trained SparseCodec-based ASR model on LibriSpeech test data.

Usage:
    python inference_cnt.py hparams/hparams.yaml --test_csv <test_csv_path> --output_folder <output_folder> [--checkpoint <ckpt_path>]

"""
import os
import torch
from speechbrain.dataio.dataloader import SaveableDataLoader, PaddedBatch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
import speechbrain as sb
from pathlib import Path
from tqdm import tqdm
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Parse arguments
import argparse
parser = argparse.ArgumentParser(description="SparseCodec ASR Inference")

parser.add_argument("model_dir", type=str, help="Path to model directory (contains hyperparams.yaml, checkpoints, test CSVs)")
parser.add_argument("--test_csv", type=str, default=None, help="Path to test CSV file (optional, overrides model_dir autodetect)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional, overrides model_dir autodetect)")
parser.add_argument("--output_folder", type=str, default=None, help="Folder to save outputs (default: model_dir)")
parser.add_argument("--num_test_pairs", type=int, default=None, help="Limit number of trial pairs loaded from veri_test.txt")
args = parser.parse_args()

# Detect paths in model_dir
model_dir = Path(args.model_dir)
hparams_path = model_dir / "hyperparams.yaml"

# Find checkpoint (latest in save/ or best in model_dir)
ckpt_path = None
if args.checkpoint is not None:
    ckpt_path = args.checkpoint
else:
    save_dir = model_dir / "save"
    if save_dir.exists():
        # Recursively search for checkpoint files in save/
        ckpts = list(save_dir.rglob("*.ckpt")) + list(save_dir.rglob("*.pth")) + list(save_dir.rglob("*.pt"))
        # Also include model.ckpt files (common SpeechBrain naming)
        ckpts += list(save_dir.rglob("model.ckpt"))
        if ckpts:
            ckpts = sorted(ckpts, key=lambda x: str(x))
            ckpt_path = str(ckpts[-1])
    if ckpt_path is None:
        # fallback: look for .ckpt/.pt in model_dir
        ckpts = sorted(model_dir.glob("*.ckpt")) + sorted(model_dir.glob("*.pth")) + sorted(model_dir.glob("*.pt"))
        if ckpts:
            ckpt_path = str(ckpts[-1])

# Find test CSV
test_csv = args.test_csv
if test_csv is None:
    # Prefer test-clean.csv, fallback to any test*.csv
    if (model_dir / "test-clean.csv").exists():
        test_csv = str(model_dir / "test-clean.csv")
    else:
        test_csvs = sorted(model_dir.glob("test*.csv"))
        if test_csvs:
            test_csv = str(test_csvs[0])
        else:
            raise FileNotFoundError("No test CSV found in model directory. Please specify with --test_csv.")


# Output folder
output_folder = args.output_folder if args.output_folder else str(model_dir)

logging.info(f"Using checkpoint: {ckpt_path}")
logging.info(f"Using test CSV: {test_csv}")
logging.info(f"Saving outputs to: {output_folder}")

with open(hparams_path, encoding="utf-8") as fin:
    hparams = load_hyperpyyaml(fin)

# Override test_csv and output_folder
hparams["test_csv"] = [test_csv]
hparams["output_folder"] = output_folder

# Tokenizer
sp_model_dir = hparams["save_folder"] if "save_folder" in hparams else hparams["output_folder"]
tokenizer = SentencePiece(
    model_dir=sp_model_dir,
    vocab_size=hparams["output_neurons"],
    annotation_train=hparams["train_csv"],
    annotation_read="wrd",
    model_type=hparams["token_type"],
    character_coverage=hparams["character_coverage"],
    bos_id=hparams["bos_index"],
    eos_id=hparams["eos_index"],
)

# Data preparation (single test set)
def dataio_prepare_infer(hparams, tokenizer, test_csv_path):
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=test_csv_path, replacements={"data_root": hparams["data_folder"]}
    )
    # Speaker label encoder
    from speechbrain.dataio.encoder import CategoricalEncoder
    spk_label_encoder = CategoricalEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    spk_label_encoder.load_or_create(path=lab_enc_file)

    # Speaker label pipeline
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        spk_id_prefix = spk_id.split("-")[0]
        yield spk_id_prefix
        spk_id_encoded = spk_label_encoder.encode_sequence_torch([spk_id_prefix], allow_unk=True)
        yield spk_id_encoded
    sb.dataio.dataset.add_dynamic_item([test_data], label_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    sb.dataio.dataset.add_dynamic_item([test_data], audio_pipeline)

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
    sb.dataio.dataset.add_dynamic_item([test_data], text_pipeline)

    # Set output keys for speaker verification
    sb.dataio.dataset.set_output_keys([test_data], ["id", "sig", "spk_id", "spk_id_encoded"])
    return test_data

test_data = dataio_prepare_infer(hparams, tokenizer, test_csv)

# Load modules
modules = hparams["modules"]
model = modules["model"] if "model" in modules else None


# Load checkpoint
if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if model is not None:
        # Try both "model_state_dict" and "state_dict"
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
else:
    print("Warning: No checkpoint found. Using randomly initialized model.")

# Inference loop


# --- Speaker verification: extract z_proj_speaker for trial pairs ---
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for mod in modules.values():
    if hasattr(mod, 'to'):
        mod.to(device)

# Load trial pairs from veri_test.txt (in model_dir or output_folder)
trial_file = None
for candidate in [os.path.join(model_dir, "veri_test.txt"), os.path.join(output_folder, "veri_test.txt")]:
    if os.path.exists(candidate):
        trial_file = candidate
        break
if trial_file is None:
    raise FileNotFoundError("Could not find veri_test.txt in model_dir or output_folder.")
logging.info(f"Using trial file: {trial_file}")



# Parse only negative trial pairs (label == '0'), limit to --num_test_pairs if set
trial_pairs = []
with open(trial_file, "r", encoding="utf-8") as f:
    for line in f:
        if args.num_test_pairs is not None and len(trial_pairs) >= args.num_test_pairs:
            break
        parts = line.strip().split()
        if len(parts) == 3:
            label, path1, path2 = parts
            if str(label) == '0':
                uttid1 = os.path.splitext(os.path.basename(path1))[0]
                uttid2 = os.path.splitext(os.path.basename(path2))[0]
                trial_pairs.append((label, uttid1, uttid2))
        else:
            continue

logging.info(f"Loaded {len(trial_pairs)} negative trial pairs from {trial_file}")
# Build a map from utt_id to batch index for fast lookup
utt_to_idx = {batch["id"][0]: i for i, batch in enumerate(SaveableDataLoader(test_data, batch_size=1))}


# Preload all utterance audio and z_proj_speaker
utt_zspk = {}
utt_spkid = {}
utt_spkid_encoded = {}
test_loader = SaveableDataLoader(test_data, batch_size=1)
for batch in tqdm(test_loader, desc="Extracting sparse codes for all utterances"):
    # batch is a dict, not a Batch object
    utt_id = batch["id"][0]
    # Move tensors to device
    batch_sig = batch["sig"].to(device)
    with torch.no_grad():
        encoder_out = modules["codec"].encoder(batch_sig.unsqueeze(1))
        _, _, h_out, _, _, _ = modules["disentangle"](encoder_out)
        utt_zspk[utt_id] = h_out.squeeze(0).cpu().numpy()
        utt_spkid[utt_id] = batch["spk_id"][0] if "spk_id" in batch else None
        utt_spkid_encoded[utt_id] = batch["spk_id_encoded"][0].cpu().numpy() if "spk_id_encoded" in batch else None



results = []

for label, utt1, utt2 in trial_pairs:
    emb1 = utt_zspk.get(utt1)
    emb2 = utt_zspk.get(utt2)
    if emb1 is None or emb2 is None:
        logging.warning(f"Missing embedding for {utt1} or {utt2}, skipping pair.")
        continue
    with torch.no_grad():
        emb1_tensor = torch.from_numpy(emb1).unsqueeze(0).to(device)
        emb2_tensor = torch.from_numpy(emb2).unsqueeze(0).to(device)
        logging.info(f"emb1_tensor shape: {emb1_tensor.shape}, emb2_tensor shape: {emb2_tensor.shape}")
        h_out = modules["disentangle"].vc_decode(emb1_tensor, emb2_tensor)
        logging.info("h_out shape: {}".format(h_out.shape))
        decoded = modules["codec"].decoder(h_out)
        logging.info("decoded shape: {}".format(decoded.shape))

   
'''
# Save trial pair embeddings as TSV
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "spk_trial_embeddings.tsv")
with open(output_path, "w", encoding="utf-8") as f:
    emb_dim = results[0]["z_proj_speaker1"].shape[0]
    header = ["label", "utt1", "utt2", "spk_id1", "spk_id2", "spk_id_encoded1", "spk_id_encoded2"] + [f"emb1_{i}" for i in range(emb_dim)] + [f"emb2_{i}" for i in range(emb_dim)]
    f.write("\t".join(header) + "\n")
    for r in results:
        row = [r["label"], r["utt1"], r["utt2"], str(r["spk_id1"]), str(r["spk_id2"]), str(r["spk_id_encoded1"]), str(r["spk_id_encoded2"])]
        row += [str(x) for x in r["z_proj_speaker1"]]
        row += [str(x) for x in r["z_proj_speaker2"]]
        f.write("\t".join(row) + "\n")
print(f"Speaker trial embeddings saved to {output_path}")
'''
