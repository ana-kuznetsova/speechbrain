import os
import dac
from tqdm import tqdm
from audiotools import AudioSignal
import numpy as np


def save_compressed_codes(in_dir, out_dir, model, n_quantizers=6):
    orig_paths = []
    for root, d, files in os.walk(in_dir):
        for f in files:
            if ".wav" or '.flac' in f:
                orig_paths.append(f"{root}/{f}")

    for f in tqdm(orig_paths):
        signal = AudioSignal(f)
        signal.to(model.device)
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x, n_quantizers=n_quantizers)
        new_f = f.replace(".wav", ".npz")
        new_f = new_f.replace(in_dir, out_dir)
        dirs = "/".join(new_f.split("/")[:-1])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        np.savez_compressed(new_f, z.detach().cpu().numpy())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_quantizers", type=int, default=12)

    args = parser.parse_args()
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.to(args.device)

    save_compressed_codes(args.in_dir, args.out_dir, model)
