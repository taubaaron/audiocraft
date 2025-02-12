from audiocraft.solvers import CompressionSolver
import torchaudio
import torch
from dora import git_save, hydra_main, XP
from audiocraft.utils import checkpoint
from audiocraft import models

def get_model(checkpoint_path, device="cpu", ):
    state = checkpoint.load_checkpoint(checkpoint_path)
    assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"
    cfg = state['xp.cfg']
    cfg.device = device
    compression_model = models.builders.get_compression_model(cfg).to(device)
    assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"

    assert 'best_state' in state and state['best_state'] != {}
    assert 'exported' not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."
    compression_model.load_state_dict(state['best_state']['model'])
    compression_model.eval()
    return compression_model

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def generate_audio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        """Generate Audio from tokens."""
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio

def main():
    print("Started inference")
    # Model
    model = CompressionSolver.model_from_checkpoint('//pretrained/facebook/encodec_32khz') # frame_rate=50hz, sample_rate=32k, n_q=4, cardinality=2048
    # model = CompressionSolver.model_from_checkpoint('//pretrained/facebook/encodec_24khz') # frame_rate=75hz, sample_rate=24k, n_q=8, cardinality=1024
    # Or load from a custom checkpoint path
    # xps_file = "01f7a36f-leaky_rely_400"
    # checkpoint_path = f'/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code/xps/{xps_file}/checkpoint.th'
    # model = get_model(checkpoint_path=checkpoint_path)
    # model = CompressionModel.get_pretrained('facebook/encodec_32khz')
    # model = CompressionModel.get_pretrained('dac_44khz')

    # Wav
    wav, sr = torchaudio.load("aaron_xai4ae/dataset/vctk_sub_dataset/p225_001_mic1.flac")  # reads audio file
    # wav, sr = torchaudio.load("/cs/labs/adiyoss/shared/data/speech/LibriSpeech/mp3/100/121669/121669.mp3")  # reads audio file
    # wav.shape
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)  # converts sample rate to match target sample rate (per model)
    wav = wav.unsqueeze(0)  # Adds another dimension for encoding function
    # wav.shape
    print("start encoding")
    audio_tokens, scale = model.encode(wav)
    print("finished encoding")
    ###
    print("start decoding")
    token_output = model.decode(audio_tokens, scale)
    print("finished decoding")

    token_output = token_output.squeeze(0)
    token_output = (token_output*32767).to(torch.int16)
    file_name = f"aaron_xai4ae/dataset/vctk_sub_dataset/p225_001_mic1-reconstructed.mp3"
    torchaudio.save(file_name, token_output, 32000)
    print(f"saved audio as: {file_name}")


if __name__ == '__main__':
    main()
    print("finished")


    """
    audio_tokens
    audio_tokens.shape
    model.sample_rate
    model.frame_rate
    model.channels
    model.frame_rate * torch.log2(torch.tensor(model.cardinality)) * model.num_codebooks  # kbps
    num_codebooks
    """
#
#
# wav = convert_audio(wav, sr, model.sample_rate, model.channels)
# from . import quantization as qt
#
# # audio_output token_output set_target_bandwidth
#
# # encoded_frames, scale = model.encode(wav)
# # codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
#
# # generated_audio = generate_audio(token_output)
# from IPython.display import display, Audio
# display(Audio(token_output.detach().cpu().squeeze(), rate=32000))




#
# x = batch.to(self.device)
# y = x.clone()
# # TODO: load noise to x from dns dataset (noise needs to be scaled + random cropped) downsample using julious
# noise, sr = torchaudio.load("/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/noises_1/audio-8khz/1-7456-A-13.wav")
#
# if noise.size(1) > x.size(1):
#     # Randomly crop noise to the length of audio
#     start = torch.randint(0, noise.size(1) - x.size(1) + 1, (1,))
#     noise = noise[:, start:start + x.size(1)]
# elif noise.size(1) < x.size(1):
#     # If noise is shorter, pad it with zeros
#     padding = x.size(1) - noise.size(1)
#     noise = torch.nn.functional.pad(noise, (0, padding))
# import random
# x = x + noise*(random.choice([2,1,0.5,0.25]))










# Finally, you can also retrieve the full Solver object, with its dataloader etc.
# from audiocraft import train
# from pathlib import Path
# import logging
# import os
# import sys
# Uncomment the following line if you want some detailed logs when loading a Solver.
# logging.basicConfig(stream=sys.stderr, level=logging.INFO)
#
# # You must always run the following function from the root directory.
# os.chdir(Path(train.__file__).parent.parent)
#
#
# # You can also get the full solver (only for your own experiments).
# # You can provide some overrides to the parameters to make things more convenient.
# solver = train.get_solver_from_sig('SIG', {'device': 'cpu', 'dataset': {'batch_size': 8}})
# solver.model
# solver.dataloaders



