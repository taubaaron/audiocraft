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

# Train
input_folder = "/cs/dataset/Download/adiyoss/valentini/8k/noisy_trainset_56spk_wav"
output_file = "/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code/egs/8k_elbit-noisy/8k_train/data.jsonl"
create_jsonl(input_folder, output_file)

# Validate
input_folder = "/cs/dataset/Download/adiyoss/valentini/8k/noisy_trainset_28spk_wav"
output_file = "/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code/egs/8k_elbit-noisy/8k_validate/data.jsonl"
create_jsonl(input_folder, output_file)

# Evaluate
input_folder = "/cs/dataset/Download/adiyoss/valentini/8k/noisy_testset_wav"
output_file = "/cs/labs/adiyoss/aarontaub/thesis/audiocraft/code/egs/8k_elbit-noisy/8k_test/data.jsonl"
create_jsonl(input_folder, output_file)