import re
import matplotlib.pyplot as plt

# Read the log file
log_file_path = "aaron_scripts/elbit_coverage/activation_comparison/output_leaky_relu2.log"  # Path to your log file
with open(log_file_path, "r") as file:
    logs = file.read()

# Extract sisnr and mel values using regex
sisnr_values = [float(match.group(1)) for match in re.finditer(r"sisnr\s*([\d\.\-]+)", logs)]
mel_values = [float(match.group(1)) for match in re.finditer(r"mel\s*([\d\.\-]+)", logs)]

# Ensure the lists are of the same length
iterations = list(range(1, min(len(sisnr_values), len(mel_values)) + 1))
sisnr_values = sisnr_values[:len(iterations)]
mel_values = mel_values[:len(iterations)]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(iterations, sisnr_values, label="sisnr", linewidth=2)
plt.plot(iterations, mel_values, label="mel", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Values")
plt.title("SISNR and Mel Values Over Iterations")
plt.legend()
plt.grid(True)

# Save the plot
output_path = "sisnr_mel_plot.png"
plt.savefig(output_path, format="png", dpi=300)
plt.close()

print(f"Plot saved as {output_path}")
