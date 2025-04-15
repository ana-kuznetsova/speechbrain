import torch
import os
import argparse
import speechbrain as sb
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import torchaudio

def dataio_prep_test(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "fold")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, fold):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/fold{:}/{:}".format(fold, wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    
    test_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams["test_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[test_dataset],
        output_key="class_string",
    )

    return test_dataset, label_encoder


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
        probs[i] = torch.bincount(codes[i].detach().cpu(), minlength=codebook_size).float()
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
        "--model_path", type=str, required=True, help="Path to the pretrained model."
    )

    args = parser.parse_args()

    # Collect audio files
    #data_folder = args.test_wav_folder
    #files = []
    #for root, d, f in os.walk(data_folder):
    #    for file in f:
    #        if file.endswith(".flac"):
    #            files.append(os.path.join(root, file))

    # Load pretrained model
    model_folder = args.model_path
    save_folder = f"{model_folder}"
    hparams_file = f"{model_folder}/../../hyperparams.yaml"

    model = EncoderClassifier.from_hparams(source=model_folder,
                                        hparams_file=hparams_file,
                                        savedir=save_folder, 
                                        run_opts={"device":"cuda"})
    model.mods.eval()

    # Load test JSON file

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin)
    dataset, label_encoder = dataio_prep_test(hparams)
    hparams["label_encoder"] = label_encoder
    
    tot_entropy = 0

    for item in tqdm(dataset):
        wav = item["sig"].unsqueeze(0).to("cuda")
        wav_len = torch.tensor([wav.shape[0]])
        feats = model.mods.compute_features(wav)
        feats = model.mods.mean_var_norm(feats, wav_len)
        z, codes = model.mods.embedding_model(feats, wav_len)
        e = compute_entropy(codes.squeeze(0), codebook_size=hparams["vocab_size"])
        tot_entropy += e
    
    print("ASR Quantized Layer Index:", hparams["quantize_layer_idx"])
    print(f"Total entropy: {tot_entropy}")
    print(f"Average entropy: {tot_entropy / len(dataset)}")
    print(f"Number of files: {len(dataset)}")
  