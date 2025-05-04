import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
import pickle

# --- Config ---
LATENT_DIR = "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/results/latent_data"
SAVE_DIR = "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/results/model"
PITCH_RANGE = list(range(-5, 6))
NUM_COMPONENTS = 30
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Latents ---
all_latents = []
valid_shapes = []
for fname in os.listdir(LATENT_DIR):
    if fname.endswith(".npz"):
        data = np.load(os.path.join(LATENT_DIR, fname))
        latents = data["latents"]
        if latents.shape == (len(PITCH_RANGE), 128):
            all_latents.append(latents)
        else:
            print(f"⚠️ Skipping {fname} due to invalid shape: {latents.shape}")

if not all_latents:
    raise ValueError("No valid latent files found!")

all_latents = np.stack(all_latents)  # (num_files, 41, 128)
num_files = all_latents.shape[0]

# --- Reshape for PCA ---
all_latents_reshaped = all_latents.reshape(-1, 128)  # (num_files*41, 128)
scaler = StandardScaler()
scaled = scaler.fit_transform(all_latents_reshaped)
pca = PCA(n_components=NUM_COMPONENTS)
pca_latents = pca.fit_transform(scaled)  # (num_files*41, num_components)

# --- Fit Splines per Component ---
splines = []
for c in range(NUM_COMPONENTS):
    component_vals = pca_latents[:, c].reshape(num_files, len(PITCH_RANGE))
    spline_c = [UnivariateSpline(PITCH_RANGE, component_vals[i], k=3, s=0)
                for i in range(num_files)]
    splines.append(spline_c)

# --- Save Models ---
with open(os.path.join(SAVE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(SAVE_DIR, "pca_model.pkl"), "wb") as f:
    pickle.dump(pca, f)
np.save(os.path.join(SAVE_DIR, "splines.npy"), splines, allow_pickle=True)

print("✅ PCA model and splines saved.")
