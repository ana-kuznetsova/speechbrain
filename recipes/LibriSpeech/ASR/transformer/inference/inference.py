import torch
import os
import argparse
import speechbrain as sb
from speechbrain.inference.ASR import EncoderDecoderASR
from tqdm import tqdm
from thop import profile, clever_format  # Import for FLOPs calculation


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

def quantizer_flops_hook(module, input, output):
    """
    Custom hook to compute FLOPs for the quantizer.

    Args:
        module (torch.nn.Module): The quantizer module.
        input (tuple): Input to the quantizer.
        output (torch.Tensor): Output from the quantizer.
    """
    input_tensor = input[0]
    batch_size, num_channels, seq_len = input_tensor.shape
    num_codebooks = module.n_codebooks
    codebook_size = module.codebook_size

    # FLOPs for quantization: searching for the closest codebook vector
    flops = batch_size * seq_len * num_codebooks * codebook_size
    module.__flops__ = flops  # Store FLOPs in the module for later retrieval

def compute_encoder_flops(model, input_shape):
    """
    Compute the number of FLOPs in the encoder of the model, including quantizer layers.

    Args:
        model (torch.nn.Module): The ASR model.
        input_shape (tuple): Shape of the input tensor (batch_size, time, features).

    Returns:
        float: Number of FLOPs in the encoder, including quantizer layers.
    """
    # Generate dummy input
    dummy_feats = torch.randn(input_shape).to(next(model.parameters()).device)
    seq_len, embed_dim = dummy_feats.shape[1], dummy_feats.shape[2]
    pos_embs = torch.randn((1, 2 * seq_len - 1, embed_dim)).to(dummy_feats.device)

    # Compute FLOPs for the encoder
    encoder = model.mods.Transformer.encoder
    flops_encoder, _ = profile(encoder, inputs=(dummy_feats, None, None, pos_embs))

    # Register the custom hook for the quantizer
    quantizer = model.mods.Transformer.encoder.quantizer
    quantizer.register_forward_hook(quantizer_flops_hook)

    # Pass dummy input through the quantizer to trigger the hook
    dummy_feats = dummy_feats.transpose(1, 2)  # Transpose to match quantizer input
    quantizer(dummy_feats)

    # Retrieve FLOPs from the quantizer
    flops_quantizer = quantizer.__flops__

    # Total FLOPs
    print(f"Encoder FLOPs: {flops_encoder / 1e9:.2f} GFLOPs")
    print(f"Quantizer FLOPs: {flops_quantizer / 1e9:.2f} GFLOPs")
    total_flops = flops_encoder + flops_quantizer
    return total_flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entropy and FLOPs.")
    parser.add_argument(
        "--test_wav_folder", type=str, required=False, help="Path to the example wav folder."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model."
    )
    parser.add_argument(
        "--vocab_size", type=int, required=False, help="Size of the codebook."
    )
    parser.add_argument(
        "--quantize_layer_idx", type=int, default=11, help="Index of the quantized layer."
    )
    parser.add_argument(
        "--compute_entropy", action="store_true", help="Flag to compute entropy of the codes."
    )
    parser.add_argument(
        "--compute_flops", action="store_true", help="Flag to compute FLOPs in the encoder."
    )
    args = parser.parse_args()

    # Load pretrained model
    model_folder = args.model_path
    save_folder = f"{model_folder}"
    hparams_file = f"{model_folder}/../../hyperparams.yaml"

    model = EncoderDecoderASR.from_hparams(source=model_folder,
                                           hparams_file=hparams_file,
                                           savedir=save_folder, 
                                           run_opts={"device":"cuda"})
    model.mods.eval()

    if args.compute_entropy:
        # Collect audio files
        data_folder = args.test_wav_folder
        files = []
        for root, d, f in os.walk(data_folder):
            for file in f:
                if file.endswith(".flac"):
                    files.append(os.path.join(root, file))
                    
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
        print("ASR Quantized Layer Index:", args.quantize_layer_idx)
        print(f"Total entropy: {tot_entropy}")
        print(f"Average entropy: {tot_entropy / len(files)}")
        print(f"Number of files: {len(files)}")

    if args.compute_flops:
        input_shape = (1, 101, 512)  # Example input shape for 1 second of audio at 16kHz
        flops = compute_encoder_flops(model, input_shape)
        print(f"Number of FLOPs in the encoder (including quantizer): {flops / 1e9:.2f} GFLOPs")