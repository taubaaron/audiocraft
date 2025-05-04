import subprocess

# --- Paths to scripts ---
SCRIPTS = [
    # "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/generate_latent_dataset.py",
    "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/fit_pitch_latent_splines.py",
    "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/apply_pitch_shift_latent.py",
]

# --- Run All ---
def main():
    for script in SCRIPTS:
        print(f"\n🚀 Running: {script}")
        subprocess.run(["python", script], check=True)

if __name__ == "__main__":
    main()
# continue