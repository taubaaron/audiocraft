import os
import torch
import torchaudio

from audiocraft.solvers import CompressionSolver

def compute_mel_spectrogram(waveform, sample_rate=32000, n_fft=1024, n_mels=128, hop_length=320):
    """
    Computes a Mel-spectrogram from the waveform.
    Args:
        waveform (torch.Tensor): Input waveform of shape (1, channels, num_samples).
    Returns:
        np.ndarray: Mel-spectrogram features.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_transform(waveform).squeeze().numpy()  # Shape (n_mels, time_frames)
    return mel_spectrogram.T  # Transpose to (time_frames, n_mels)

def run_inference(model, input_path, output_dir):
    """
    Runs EnCodec inference on a single file, extracting RVQ tokens and decoding audio.
    Saves the input features, tokens, and reconstructed audio for reference.
    """
    # Load and preprocess audio
    wav, sr = torchaudio.load(input_path)
    wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)
    wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
    wav = wav.unsqueeze(0)  # Add batch dimension

    print(f"Encoding {input_path}...")
    audio_tokens, _ = model.encode(wav)

    # Decode audio for reference
    print("Decoding audio...")
    reconstructed_audio = model.decode(audio_tokens).squeeze(0)
    reconstructed_audio = (reconstructed_audio * 32767).to(torch.int16)  # Convert to 16-bit PCM

    # Ensure reconstructed audio has 2D shape for torchaudio.save
    if reconstructed_audio.dim() == 1:
        reconstructed_audio = reconstructed_audio.unsqueeze(0)

    # Compute Mel-spectrogram
    mel_spec = compute_mel_spectrogram(wav, sample_rate=model.sample_rate)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).replace(".flac", "")
    torch.save(mel_spec, os.path.join(output_dir, f"{base_name}_mel_spec.pt"))
    torch.save(audio_tokens, os.path.join(output_dir, f"{base_name}_tokens.pt"))
    torchaudio.save(os.path.join(output_dir, f"{base_name}_reconstructed.wav"), reconstructed_audio, model.sample_rate)

    print(f"Saved Mel-spectrogram, tokens, and reconstructed audio for {input_path}.")

def batch_inference(input_dir, output_dir, model):
    """
    Processes all FLAC files in a directory through EnCodec inference.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                input_path = os.path.join(root, file)
                run_inference(model, input_path, output_dir)

def main_inference():
    print("Starting batch inference...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/results/inference_outputs"
    batch_inference(input_dir, output_dir, model)
    print("Batch inference complete.")

if __name__ == "__main__":
    main_inference()