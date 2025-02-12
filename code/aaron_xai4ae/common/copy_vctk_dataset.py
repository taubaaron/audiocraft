import os
import shutil

# Define source and destination directories
source_dir = "/cs/labs/adiyoss/shared/data/speech/vctk/wav32_silence_trimmed"  # Replace with actual source directory
destination_dir = "aaron_xai4ae/common/dataset/vctk_sub_dataset"
destination_dir = "aaron_xai4ae/approach_3/results/vctk_sub_dataset-attempt_2"

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Define speakers or files to copy (e.g., speaker IDs or specific files)
selected_speakers = ["p236", "p238", "p248", "p364", "p374", "p376"]  # Replace with desired speaker IDs

# Copy relevant files
for speaker in selected_speakers:
    speaker_dir = os.path.join(source_dir, speaker)
    print(f"Working on {speaker}")
    if os.path.exists(speaker_dir):
        dest_speaker_dir = os.path.join(destination_dir, speaker)
        os.makedirs(dest_speaker_dir, exist_ok=True)
        for file_name in os.listdir(speaker_dir):
            if file_name.endswith(".flac"):  # Only copy audio files
                source_file = os.path.join(speaker_dir, file_name)
                dest_file = os.path.join(dest_speaker_dir, file_name)
                shutil.copy2(source_file, dest_file)  # Preserve metadata
        print(f"Copied files for speaker: {speaker}")
    else:
        print(f"Speaker directory not found: {speaker}")

print("File copying completed!")
