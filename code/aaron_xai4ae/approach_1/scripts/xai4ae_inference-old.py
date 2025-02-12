import os
import torchaudio
import torch
from audiocraft.solvers import CompressionSolver
from audiocraft.utils import checkpoint
from audiocraft import models
from dora import git_save, hydra_main, XP

def get_model(checkpoint_path, device="cpu"):
    """
    Loads a pretrained EnCodec model from a checkpoint.
    """
    state = checkpoint.load_checkpoint(checkpoint_path)
    assert state is not None and 'xp.cfg' in state, f"Could not load model from checkpoint: {checkpoint_path}"
    cfg = state['xp.cfg']
    cfg.device = device
    compression_model = models.builders.get_compression_model(cfg).to(device)
    compression_model.load_state_dict(state['best_state']['model'])
    compression_model.eval()
    return compression_model

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    """
    Resamples and converts audio to the target sample rate and channel count.
    """
    if wav.shape[0] > target_channels:
        wav = wav.mean(0, keepdim=True)  # Convert to mono if target_channels is 1
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def process_audio_file(model, input_path, output_path):
    """
    Processes a single audio file through the model (encode -> decode) and saves the output.
    """
    wav, sr = torchaudio.load(input_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)  # Add batch dimension

    print(f"Encoding {input_path}...")
    audio_tokens, scale = model.encode(wav)
    print(f"Decoding {input_path}...")
    reconstructed_audio = model.decode(audio_tokens, scale).squeeze(0)

    reconstructed_audio = (reconstructed_audio * 32767).to(torch.int16)
    torchaudio.save(output_path, reconstructed_audio, model.sample_rate)
    print(f"Saved reconstructed audio: {output_path}")

def batch_process_audio(model, input_dir, output_dir):
    """
    Processes all audio files in a directory through the model and saves outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path.replace(".flac", "-reconstructed.wav"))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                process_audio_file(model, input_path, output_path)

def main():
    print("Starting batch inference...")
    # Load pretrained model
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    # Load trained model
    # checkpoint_path = f'/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code/xps/{xps_file}/checkpoint.th'
    # model = get_model(checkpoint_path=checkpoint_path)

    # Directories
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/dataset/vctk_reconstructed"

    # Batch process all files
    batch_process_audio(model, input_dir, output_dir)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
