import os
import json
import librosa
import torchaudio


def get_audio_info(audio_path):
    # Get audio duration and sample rate
    info = {}
    try:
        info = {
            "duration": librosa.get_duration(path=audio_path),
            "sample_rate": torchaudio.info(audio_path).sample_rate
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    return info

def create_jsonl(input_folder, output_file):
    # Open JSONL file for writing
    with open(output_file, 'w') as f_out:
        # Iterate through subfolders
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac')):  # Add more audio formats if needed
                    audio_path = os.path.join(root, file)
                    audio_info = get_audio_info(audio_path)
                    json_line = {
                        "path": audio_path,
                        "duration": audio_info.get("duration", None),
                        "sample_rate": audio_info.get("sample_rate", None),
                        "amplitude": None,
                        "weight": None,
                        "info_path": None
                    }
                    f_out.write(json.dumps(json_line) + '\n')

# Example usage
input_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-valid"
output_file = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-valid/data.jsonl"

create_jsonl(input_folder, output_file)
