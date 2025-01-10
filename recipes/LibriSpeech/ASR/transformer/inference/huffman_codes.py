import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from tqdm import tqdm
import os
import argparse
from collections import Counter, defaultdict
import heapq


def calculate_vector_frequencies(vectors):
    """
    Calculate the frequency of each unique vector in the input data.

    Args:
        vectors (torch.Tensor): Input data as a tensor of shape [num_vectors, vector_dim].

    Returns:
        dict: A dictionary with vectors as keys and their frequencies as values.
    """
    vectors_list = [tuple(vec.tolist()) for vec in vectors]
    return Counter(vectors_list)


class HuffmanNode:
    def __init__(self, vector, freq):
        self.vector = vector
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    """
    Build a Huffman tree based on vector frequencies.

    Args:
        frequencies (dict): A dictionary with vectors as keys and their frequencies as values.

    Returns:
        HuffmanNode: The root node of the Huffman tree.
    """
    heap = [HuffmanNode(vector, freq) for vector, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]


def generate_huffman_codes(node, prefix="", codebook=None):
    """
    Generate Huffman codes from the Huffman tree.

    Args:
        node (HuffmanNode): The root node of the Huffman tree.
        prefix (str): The current prefix for the Huffman code.
        codebook (dict): A dictionary to store the generated Huffman codes.

    Returns:
        dict: A dictionary with vectors as keys and their Huffman codes as values.
    """
    if codebook is None:
        codebook = {}

    if node.vector is not None:
        codebook[node.vector] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)

    return codebook


def huffman_encode_vectors(vectors, codebook):
    """
    Encode the input vectors using the generated Huffman codes.

    Args:
        vectors (torch.Tensor): Input data as a tensor of shape [num_vectors, vector_dim].
        codebook (dict): A dictionary with vectors as keys and their Huffman codes as values.

    Returns:
        str: The encoded data as a string of bits.
    """
    vectors_list = [tuple(vec.tolist()) for vec in vectors]
    return "".join(codebook[vector] for vector in vectors_list)


def huffman_decode_vectors(encoded_data, root):
    """
    Decode the encoded data back to the original vectors.

    Args:
        encoded_data (str): The encoded data as a string of bits.
        root (HuffmanNode): The root node of the Huffman tree.

    Returns:
        torch.Tensor: The decoded data as a tensor of shape [num_vectors, vector_dim].
    """
    decoded_vectors = []
    node = root
    for bit in encoded_data:
        if bit == "0":
            node = node.left
        else:
            node = node.right

        if node.vector is not None:
            decoded_vectors.append(node.vector)
            node = root

    return torch.tensor(decoded_vectors)


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
        "--model_folder",
        type=str,
        help="The folder containing the model.",
        required=True,
    )
    parser.add_argument(
        "--frames_per_second",
        type=int,
        default=25,
        help="Frame rate of the model",
    )

    parser.add_argument("--compute_entropy", action="store_true")
    parser.add_argument("--compute_codebook", action="store_true")
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--num_codebooks", type=int, default=2)
    parser.add_argument("--num_samples", type=int)

    args = parser.parse_args()

    save_folder = f"{args.model_folder}"
    hparams_file = f"{args.model_folder}/hyperparams.yaml"

    model = EncoderDecoderASR.from_hparams(
        source=args.model_folder, hparams_file=hparams_file, savedir=save_folder
    )
    model.mods.eval()

    # Collect files
    files = []
    for root, d, f in os.walk(args.data_folder):
        for file in f:
            if file.endswith(".flac") and "train" in root:
                files.append(os.path.join(root, file))
    print(f"Found {len(files)} files")
    if  args.num_samples:
        files = files[: args.num_samples]

    # Compute vector frequencies
    code_vector_freqs = torch.zeros(args.num_codebooks, args.codebook_size)

    for f in tqdm(files):
        aud = model.load_audio(f)
        feats = model.hparams.compute_features(aud.unsqueeze(0))
        feat_lens = torch.tensor([feats.shape[1]])
        feats = model.mods.CNN(feats)
        z, codes = model.encode_batch(feats, feat_lens)
        codes = codes.squeeze(0)
        # Compute frequencies for each codebook
        for i in range(args.num_codebooks):
            code_vector_freqs[i] += torch.histc(
                codes[i].float(),
                bins=args.codebook_size,
                min=0,
                max=args.codebook_size - 1,
            )


    # Build Huffman tree for each of the codebooks
    codebooks = {i:None for i in range(args.num_codebooks)}
    for i in range(args.num_codebooks):
        freq_dict = {j:code_vector_freqs[i][j].item() for j in range(args.codebook_size) if code_vector_freqs[i][j] != 0}
        root = build_huffman_tree(freq_dict)
        codebook_i = generate_huffman_codes(root)
        codebooks[i] = codebook_i
    torch.save(codebooks, f"{save_folder}/codebooks.pt")
