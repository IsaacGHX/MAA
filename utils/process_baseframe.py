import os
import re
import csv
import glob

# --- Configuration ---
BASE_DIR = '../output/baseframe'  # Root directory of log files
LOGS_SUBDIR = 'logs'      # Name of the subdirectory containing log files
OUTPUT_CSV = 'consolidated_test_metrics.csv' # Output CSV filename
# --- End Configuration ---

def parse_metrics_line(line):
    """Parse metric values from a log line"""
    # Use regex to find metric=value pairs
    # This pattern assumes metrics appear in the order MSE, MAE, RMSE, MAPE, and allows for other characters in between
    pattern = re.compile(
        r"MSE=([\d\.]+).*"
        r"MAE=([\d\.]+).*"
        r"RMSE=([\d\.]+).*"
        r"MAPE=([\d\.]+)"
    )
    match = pattern.search(line)
    if match:
        # Return a dictionary containing metric values
        return {
            'MSE': match.group(1),
            'MAE': match.group(2),
            'RMSE': match.group(3),
            'MAPE': match.group(4),
        }
    else:
        return None

def find_and_process_logs(base_dir, logs_subdir, output_csv):
    """Find log files, process them, and generate a CSV"""
    results = []
    # Build the pattern to find log files
    # e.g., ./baseframe/*/ */logs/*.log
    log_pattern = os.path.join(base_dir, '*', '*', logs_subdir, '*.log')
    log_files = glob.glob(log_pattern)

    print(f"Found {len(log_files)} log files to process...")

    for log_file_path in log_files:
        try:
            # Extract dataset name and model name from the path
            # Path format: ./baseframe/{dataset_name}/{model_name}/logs/{filename}.log
            parts = log_file_path.split(os.sep)
            if len(parts) >= 5 and parts[-2] == logs_subdir:
                model_name = parts[-3]
                dataset_name = parts[-4]
            else:
                print(f"Warning: Could not parse dataset/model name from path: {log_file_path}")
                continue

            last_test_line = None
            # Read the file and find the last line containing "Test Metrics"
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # More precise match, ensuring it's a test metrics line
                    if "INFO" in line and "Test Metrics" in line:
                        last_test_line = line.strip() # Save the last line found

            if last_test_line:
                metrics = parse_metrics_line(last_test_line)
                if metrics:
                    # Add dataset and model names to results
                    metrics['Dataset'] = dataset_name
                    metrics['Model'] = model_name
                    results.append(metrics)
                else:
                    print(f"Warning: Could not parse metrics line: {last_test_line} (from file: {log_file_path})")
            else:
                 print(f"Warning: No 'Test Metrics' line found in file: {log_file_path}")

        except FileNotFoundError:
            print(f"Error: File not found: {log_file_path}")
        except Exception as e:
            print(f"Error processing file {log_file_path}: {e}")

    # Write to CSV file
    if not results:
        print("No results found to write to CSV.")
        return

    # Define the order of CSV headers
    headers = ['Dataset', 'Model', 'MSE', 'MAE', 'RMSE', 'MAPE']

    print(f"Writing {len(results)} results to {output_csv}...")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader() # Write header row
            writer.writerows(results) # Write all result rows
        print(f"Successfully created CSV file: {output_csv}")
    except IOError as e:
        print(f"Error writing CSV file {output_csv}: {e}")
    except Exception as e:
         print(f"Unknown error occurred when writing CSV: {e}")

# --- Script Execution ---
if __name__ == "__main__":
    find_and_process_logs(BASE_DIR, LOGS_SUBDIR, OUTPUT_CSV)