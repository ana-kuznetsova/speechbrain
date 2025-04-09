import torch
import os
import argparse
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Compute entropy of codes.")
    parser.add_argument(
        "--test_wav_folder", type=str, required=True, help="Path to the example wav folder."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model."
    )
    parser.add_argument(
        "--vocab_size", type=int, required=True, help="Size of the codebook."
    )
    args = parser.parse_args()

    # Collect audio files
    data_folder = args.test_wav_folder
    files = []
    for root, d, f in os.walk(data_folder):
        for file in f:
            if file.endswith(".flac"):
                files.append(os.path.join(root, file))

    # Load pretrained model
    model_folder = args.model_path
    save_folder = f"{model_folder}/inference"
    hparams_file = f"{model_folder}/../../hyperparams.yaml"

    model = EncoderDecoderASR.from_hparams(source=model_folder,
                                        hparams_file=hparams_file,
                                        savedir=save_folder, 
                                        run_opts={"device":"cuda"})
    model.mods.eval()

    tot_entropy = 0

    for f in tqdm(files):
        aud = model.load_audio(f)
        feats = model.hparams.compute_features(aud.unsqueeze(0))
        feat_lens = torch.tensor([feats.shape[1]])
        feats = model.mods.CNN(feats)
        z, codes = model.encode_batch(feats, feat_lens)
        
        e = compute_entropy(
            codes.squeeze(0), codebook_size=args.vocab_size
        )
        tot_entropy += e
    print(f"Total entropy: {tot_entropy}")
    print(f"Average entropy: {tot_entropy / len(files)}")
    print(f"Number of files: {len(files)}")