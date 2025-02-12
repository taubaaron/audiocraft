import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from audiocraft.solvers import CompressionSolver

class EnCodecRVQWrapper:
    """
    A wrapper around the EnCodec model to focus on input-to-RVQ mapping.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        # Ensure input is a tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        if inputs.dim() == 2:  # Shape is (time_frames, n_mels)
            inputs = inputs.unsqueeze(0)  # Add batch dimension
        elif inputs.dim() == 3 and inputs.shape[1] > 2:  # Handle invalid channel dimensions
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {inputs.shape[1]}")
        with torch.no_grad():
            audio_tokens, _ = self.model.encode(inputs)
        return audio_tokens.cpu().numpy()

def run_shap_analysis(model, input_features, tokens, output_dir, file_name_prefix):
    """
    Runs SHAP analysis for the given Mel-spectrogram and tokens, and saves the results.
    """
    print(f"Running SHAP for: {file_name_prefix}...")

    background_data = input_features[:10]  # First 10 frames as background
    background_data = np.expand_dims(background_data, axis=1)  # Shape: (10, 1, 128)


    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model, background_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_features)

    # Save SHAP values
    shap_output_path = os.path.join(output_dir, f"{file_name_prefix}_shap_values.pt")
    torch.save(shap_values, shap_output_path)

    # Visualize SHAP importance as a spectrogram
    influence_map = np.mean(shap_values, axis=0)  # Aggregate SHAP values over outputs
    plt.imshow(influence_map, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(label="SHAP Influence")
    plt.title("SHAP Influence Map")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency Bin")
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_shap_influence_map.png"))
    plt.close()
    print(f"Saved SHAP influence map for {file_name_prefix}.")

def run_analysis(input_dir, output_dir, model):
    """
    Runs SHAP analysis for all files in the input directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_mel_spec.pt"):
                mel_spec = torch.load(os.path.join(root, file))
                tokens = torch.load(os.path.join(root, file.replace("_mel_spec.pt", "_tokens.pt")))
                file_name_prefix = file.replace("_mel_spec.pt", "")
                run_shap_analysis(wrapper, mel_spec, tokens, output_dir, file_name_prefix)

def main_shap_analysis():
    print("Starting SHAP analysis...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/results/inference_outputs"
    output_dir = "aaron_xai4ae/results/shap_outputs"
    os.makedirs(output_dir, exist_ok=True)

    global wrapper
    wrapper = EnCodecRVQWrapper(model)

    run_analysis(input_dir, output_dir, model)

    print("SHAP analysis complete.")

if __name__ == "__main__":
    main_shap_analysis()
