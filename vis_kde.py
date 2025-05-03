import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import glob

def plot_true_pred_density(df, output_dir, filename, alpha=0.5, no_grid=False):
    """
    Plot the density estimate for true and predicted values for each CSV file.
    Args:
        df (pd.DataFrame): DataFrame containing 'true' and 'pred' columns.
        output_dir (str): Directory to save the plots.
        filename (str): Original CSV filename (without extension).
        alpha (float): Transparency of the filled area.
        no_grid (bool): Whether to remove grid lines.
    """
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))

    # Plot KDE area plots for true and predicted values
    sns.kdeplot(df['true'].dropna(), label='True', color='orange',
                linewidth=0.8, alpha=alpha, fill=True)  # Change linewidth
    sns.kdeplot(df['pred'].dropna(), label='Pred', color='blue',
                linewidth=0.8, alpha=alpha, fill=True)  # Change linewidth

    # plt.title(f'Density: True vs Pred ({filename})', fontsize=16)
    ax = plt.gca()
    ax.set(xlabel=None, ylabel=None)
    # plt.xlabel('Value', fontsize=14)
    # plt.ylabel('Density', fontsize=14)
    plt.legend()
    if not no_grid:
        plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{filename}_true_pred_density.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density estimates of true and predicted values for each CSV file.')
    parser.add_argument('--input_dir', type=str, default='true2pred',
                        help='Directory containing CSV files.')
    parser.add_argument('--output_dir', type=str, default='outputs_vis',
                        help='Directory to save the plots.')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency of the filled area.')
    parser.add_argument('--no_grid', action='store_true', default=True,
                        help='Add this parameter to remove grid lines.') # Use action='store_true' for flags

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = glob.glob(os.path.join(args.input_dir, '*.csv'))
    if not csv_paths:
        print(f"No CSV files found in directory {args.input_dir}.")
        exit(1)

    # --- Start of modification ---

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))  # Create one figure outside the loop

    all_true_series = []
    pred_series_list = []
    pred_labels = []

    # First loop: Read all files, collect data
    print("Reading and collecting data...")
    for path in csv_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Could not read file {path}: {e}")
            continue

        if 'true' not in df.columns or 'pred' not in df.columns:
            print(f"File {path} is missing 'true' or 'pred' columns, skipping.")
            continue

        # Collect true and predicted value data
        all_true_series.append(df['true'].dropna())
        pred_series_list.append(df['pred'].dropna())
        pred_labels.append(filename)  # Use filename as the label for the prediction distribution

    if not all_true_series:
        print("No valid data found in any file.")
        plt.close()  # Close the empty figure created earlier
        exit(1)

    # Combine all true value data and plot their overall density distribution
    combined_true = pd.concat(all_true_series).dropna()
    if not combined_true.empty:
        # Use the nicer style you mentioned: thin border (linewidth=1.5), semi-transparent fill (alpha=args.alpha)
        sns.kdeplot(combined_true, label='True (Combined)', color='orange',
                    linewidth=1.5, alpha=args.alpha, fill=True)
    else:
        print("No valid true data found, skipping plotting True distribution.")

    # Plot the predicted value density distribution for each file
    print("Plotting all prediction distributions...")
    # Seaborn automatically chooses colors for different curves
    # If you want to control colors, you can use seaborn.color_palette or specify manually
    for pred_series, label in zip(pred_series_list, pred_labels):
        if not pred_series.empty:
            # Also use thin border and semi-transparent fill
            sns.kdeplot(pred_series, label=f'Pred ({label})',
                        linewidth=1.5, alpha=args.alpha, fill=True)  # Let seaborn automatically choose colors
        else:
            print(f"No valid predicted data found in file {label}, skipping plotting.")

    # --- Finish plotting setup ---
    ax = plt.gca()
    ax.set(xlabel='Value', ylabel='Density')  # Add axis labels
    # Optionally add a main title
    plt.title('Density Distribution: Combined True vs All Predictions', fontsize=16)

    plt.legend()  # Add legend, showing the filename for each prediction
    if not args.no_grid:
        plt.grid(True, linestyle='--', alpha=0.6)  # Can make grid lines softer

    plt.tight_layout()

    # Save the plot once outside the loop
    out_path = os.path.join(args.output_dir, 'all_predictions_combined_density.png')
    try:
        plt.savefig(out_path)
        print(f"Saved combined plot: {out_path}")
    except Exception as e:
        print(f"Could not save plot {out_path}: {e}")

    plt.close()  # Close the figure