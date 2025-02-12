import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def parse_log_file(file_path):
    """
    Parse the log file to extract epoch summaries and metrics.
    """
    data = []
    # Updated pattern to match any line containing "Epoch"
    epoch_pattern = r".*?Epoch (\d+).*?\|?(.*)"

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(epoch_pattern, line)
            if match:
                epoch = int(match.group(1))  # Extracts epoch number
                metrics = match.group(2).strip()  # Extracts the rest of the metrics

                metrics_dict = {"Epoch": epoch}

                for metric in metrics.split(" | "):
                    if "=" in metric:
                        key, value = metric.split("=", 1)
                        try:
                            # Attempt to convert the value to float after stripping ANSI escape codes
                            value = re.sub(r"\x1b\[[^m]*m", "", value).strip()
                            metrics_dict[key.strip()] = float(value)
                        except ValueError:
                            # If conversion fails, store the raw value for debugging
                            metrics_dict[key.strip()] = re.sub(r"\x1b\\[.*?m", "", value).strip()

                data.append(metrics_dict)

    return pd.DataFrame(data)

def plot_metrics(df, metrics, title="Training and Validation Metrics", save_path=None):
    """
    Plot specified metrics for training and validation phases.
    """
    plt.figure(figsize=(12, 8))
    for metric in metrics:
        if metric in df.columns:
            plt.plot(df['Epoch'], df[metric], label=metric)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def clean_column_names(df):
    # Clean up column names
    df.columns = df.columns.str.strip().str.replace(r'\|', '', regex=True).str.replace(' ', '')

def compare_activation_functions(parsed_csv_files, metric, save_path=None):
    """
    Compare the performance of different activation functions based on a specific metric.
    """
    plt.figure(figsize=(12, 8))
    for csv_file in parsed_csv_files:
        activation = os.path.basename(csv_file).replace("output_relu_parsed.csv", "")
        try:
            # Adjust delimiter as needed
            df = pd.read_csv(csv_file, delimiter='|', on_bad_lines='skip', skip_blank_lines=True)
            clean_column_names(df)

            print(f"Columns in {csv_file}: {df.columns.tolist()}")  # Debugging step

            if metric in df.columns and df[metric].notna().any():
                plt.plot(df['Epoch'], df[metric], label=f"{activation}")
            else:
                print(f"Warning: Metric '{metric}' not found or all NaN in {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Comparison of Activation Functions for {metric}")
    if len(plt.gca().get_lines()) > 0:
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        print("No valid data to plot.")

if __name__ == "__main__":
    os.chdir("./aaron_scripts/elbit_coverage/activation_comparison/")

    # Parse all log files in the current directory
    log_files = glob.glob("*.log")
    parsed_csv_files = []

    # Process each log file and save individual analyses
    for log_file in log_files:
        df = parse_log_file(log_file)
        activation = os.path.basename(log_file).replace(".log", "")
        csv_file = f"{activation}_parsed.csv"
        df.to_csv(csv_file, index=False)
        parsed_csv_files.append(csv_file)



        # Plot metrics for each log file
        plot_metrics(df, metrics=["sisnr", "mel"], title=f"Metrics for {activation}", save_path=f"{activation}_metrics.png")

    # Compare all activation functions based on a chosen metric
    compare_activation_functions(log_files, metric="sisnr", save_path="comparison_sisnr.png")

if __name__ == "__main__":
    os.chdir("./aaron_scripts/elbit_coverage/activation_comparison/")
    # Parse all log files in the current directory
    log_files = glob.glob("*.log")
    parsed_csv_files = []

    # Process each log file and save individual analyses
    for log_file in log_files:
        df = parse_log_file(log_file)
        activation = os.path.basename(log_file).replace(".log", "")
        csv_file = f"{activation}_parsed.csv"
        df.to_csv(csv_file, index=False)
        parsed_csv_files.append(csv_file)

        # Plot metrics for each log file
        plot_metrics(df, metrics=["sisnr", "mel"], title=f"Metrics for {activation}", save_path=f"{activation}_metrics.png")

    # Compare all activation functions based on a chosen metric
    compare_activation_functions(parsed_csv_files, metric="sisnr", save_path="comparison_sisnr.png")


