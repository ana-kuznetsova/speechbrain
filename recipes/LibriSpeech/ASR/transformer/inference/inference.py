import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from tqdm import tqdm
import os
import random
import argparse
import matplotlib.pyplot as plt


def compute_entropy(codes: torch.Tensor, codebook_size: int):
    """
    Compute entropy of the codes.

    Args:
        codes (torch.Tensor): [num_codebooks, T] tensor of codes.
        codebook_size (int): Size of the codebook.

    Returns:
        float: Entropy of the codes.
    """

    num_codebooks, T = codes.shape
    probs = torch.zeros(num_codebooks, codebook_size).float()
    for i in range(num_codebooks):
        probs[i] = torch.histc(
            codes[i].float(), bins=codebook_size, min=0, max=codebook_size - 1
        )

    # Normalize to get probabilities
    probs = probs / T

    # Add a small constant to avoid log(0)
    epsilon = 1e-10

    probs = torch.where(probs > 0, probs, epsilon)

    entropy = torch.zeros(num_codebooks)
    for i in range(num_codebooks):
        entropy[i] = -torch.sum(probs[i] * torch.log2(probs[i]))

    return entropy.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot entropy of audio files."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="The root directory to search for audio files.",
        required=True,
    )
    parser.add_argument(
        "--model_folder", type=str, help="The folder containing the model.",
        required=True,
    )
    parser.add_argument(
        "--frames_per_second",
        type=int,
        default=25,
        help="Frame rate of the model",
    )

    parser.add_argument("--compute_entropy", action="store_true")
    parser.add_argument("--code_distribution", action="store_true")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--num_codebooks", type=int, default=2)
    parser.add_argument("--num_samples", type=int)

    args = parser.parse_args()

    save_folder = f"{args.model_folder}/inference"
    hparams_file = f"{args.model_folder}/hyperparams.yaml"

    model = EncoderDecoderASR.from_hparams(
        source=args.model_folder, hparams_file=hparams_file, savedir=save_folder
    )
    model.mods.eval()

    # Collect files
    files = []
    for root, d, f in os.walk(args.data_folder):
        for file in f:
            if file.endswith(".flac"):
                files.append(os.path.join(root, file))

    random.shuffle(files)
    if args.num_samples:
        files = files[:args.num_samples]
    tot_entropy = 0
    probs = torch.zeros(args.num_codebooks, args.codebook_size).float()

    for f in tqdm(files):
        aud = model.load_audio(f)
        feats = model.hparams.compute_features(aud.unsqueeze(0))
        feat_lens = torch.tensor([feats.shape[1]])
        feats = model.mods.CNN(feats)
        z, codes = model.encode_batch(feats, feat_lens)
        if args.compute_entropy:
            e = compute_entropy(
                codes.squeeze(0), codebook_size=args.codebook_size
            )
            tot_entropy += e

        if args.code_distribution:
            for i in range(args.num_codebooks):
                codes = codes.squeeze(0)
                probs[i] = torch.histc(
                    codes[i].float(),
                    bins=args.codebook_size,
                    min=0,
                    max=args.codebook_size - 1,
                )

    if args.compute_entropy:
        frames_per_second = 25
        tot_entropy = tot_entropy / len(files)
        print(f"Average entropy: {tot_entropy}")
        with open(f"{args.model_folder}/entropy.txt", "w") as f:
            f.write(f"Avg entropy: {tot_entropy}\n")
            bitrate = frames_per_second * tot_entropy
            f.write(f"Bitrate: {bitrate}")

    if args.code_distribution:
        probs = probs / len(files)
        fig, axes = plt.subplots(
            args.num_codebooks, 1, figsize=(9 * args.num_codebooks, 8)
        )
        axes = axes.ravel()
        for i in range(args.num_codebooks):
            axes[i].bar(
                list(range(args.codebook_size)), probs[i].numpy(), edgecolor="r"
            )
            axes[i].set_title(f"Codebook {i}")
            axes[i].grid(color="grey", linestyle="--", linewidth=0.5)
        fig.suptitle(
            f"Code distribution for num_codebooks={args.num_codebooks}, codebook_size={args.codebook_size}"
        )
        plt.savefig(f"{args.model_folder}/code_distribution.png")
