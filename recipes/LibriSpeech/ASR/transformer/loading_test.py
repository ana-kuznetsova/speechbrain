import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from hyperpyyaml import load_hyperpyyaml

if __name__=="__main__":
    hparams_file = '/data/anakuzne/speechbrain/recipes/LibriSpeech/ASR/transformer/hparams/conformer_large_quantizer.yaml'
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    ckpt_save_dir = '/data/anakuzne/speechbrain/recipes/LibriSpeech/ASR/transformer/results/conformer_large_quantizer_pretrain_codebook_only/3407/save/CKPT+2024-09-19+21-56-14+00/'
    asr_model = EncoderDecoderASR.from_hparams(source=ckpt_save_dir, hparams_file=hparams_file)