import dac
import torch
import torch.nn as nn
import speechbrain as sb

class SpeechCodes(torch.nn.Module):
    """Generate features from pretrained neural codec.

    Arguments:
    ----------
        encodec_path : str (default: None)
            path to the encodec ckpt if downloaded
        model_sample_rate : str (default: 16khz)
            sample rate for speech codec. Avail 16khz, 44khz
        wav_sample_rate : int (default: 16000)
            data sample rate
        device : str (default: "cuda")
            cuda or cpu
        trainable : bool (default: False)
            whether the encodec should be frozen
    """
    def __init__(self,
                 encodec_path = None,
                 model_sample_rate = "16khz",
                 wav_sample_rate = 16000,
                 device = "cuda",
                 trainable = False):
        super().__init__()
        if not encodec_path:
            encodec_path = dac.utils.download(model_type = model_sample_rate)
        self.encodec_path = encodec_path
        self.enc_model = dac.DAC.load(self.encodec_path).to(device)
        if not trainable:
            self.enc_model.eval()
        self.wav_sample_rate = wav_sample_rate

    def forward(self, wav):
        """Get speech codes for feature extraction.
        """
        assert len(wav.shape) == 2, "Wav inp shape must be 2D"
        wav = wav.unsqueeze(1)
        wav = self.enc_model.preprocess(wav, self.wav_sample_rate)
        z_feats, _, _, _, _ = self.enc_model.encode(wav)
        return z_feats