import logging
import os
import torchaudio
import torch
from dora import git_save, hydra_main
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.utils.checkpoint import resolve_checkpoint_path

logger = logging.getLogger(__name__)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

@hydra_main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg):
    logger.info("Started inference")

    # Load checkpoint
    checkpoint_path = "xps/01f7a36f/checkpoint_400.th"
    logger.info(f"Using checkpoint: {checkpoint_path}")
    _checkpoint_path = resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
    if not os.path.exists(_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {_checkpoint_path}")

    # Load model
    model = CompressionSolver.model_from_checkpoint(_checkpoint_path)
    logger.info("Model loaded successfully!")

    # Load and preprocess audio
    input_audio_path = "code/dataset/example/electro_1.mp3"
    wav, sr = torchaudio.load(input_audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)

    # Encoding
    logger.info("Start encoding")
    audio_tokens, scale = model.encode(wav)
    logger.info("Finished encoding")

    # Decoding
    logger.info("Start decoding")
    token_output = model.decode(audio_tokens, scale)
    logger.info("Finished decoding")

    # Save the reconstructed audio
    token_output = token_output.squeeze(0).mul(32767).to(torch.int16)
    output_audio_path = "code/dataset/example/electro_1-reconstructed.mp3"
    torchaudio.save(output_audio_path, token_output, model.sample_rate)
    logger.info(f"Saved reconstructed audio to: {output_audio_path}")

if __name__ == "__main__":
    main()
