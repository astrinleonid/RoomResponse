import numpy as np
import matplotlib.pyplot as plt


def plot_array_lines(data_array, line_indices=None, labels=None, title="Data from Files",
                     xlabel="Index", ylabel="Value", figsize=(10, 6)):
    """
    Plots selected rows of a NumPy array as separate lines.

    Args:
        data_array (numpy.ndarray): 2D NumPy array where each row is plotted as a line
        line_indices (list, optional): Indices of the rows to plot. If None, plots all rows
        labels (list, optional): List of labels for each line (defaults to "File 1", "File 2", etc.)
        title (str, optional): Plot title
        xlabel (str, optional): X-axis label
        ylabel (str, optional): Y-axis label
        figsize (tuple, optional): Figure size (width, height) in inches

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # If no line indices are provided, use all rows
    if line_indices is None:
        line_indices = range(data_array.shape[0])

    # If no labels are provided, create default ones
    if labels is None:
        labels = [f"File {i + 1}" for i in range(data_array.shape[0])]

    # Plot each selected row as a line
    for idx in line_indices:
        if idx < 0 or idx >= data_array.shape[0]:
            print(f"Warning: Index {idx} is out of bounds. Skipping.")
            continue

        row = data_array[idx]
        # Check for NaN values (if we used padding)
        mask = ~np.isnan(row)
        if np.any(mask):  # Only plot if there are non-NaN values
            ax.plot(np.arange(len(row))[mask], row[mask], label=labels[idx])

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add legend if we have multiple lines
    if len(line_indices) > 1:
        ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

# Example usage:
# matching_files = extract_files_with_substring('/path/to/folder', 'data')
# data_array = combine_files_to_numpy_array(matching_files)
#
# # Use file names as labels
# file_names = [os.path.basename(f) for f in matching_files]
#
# # Plot only lines 0, 2, and 5 (first, third, and sixth files)
# fig = plot_array_lines(data_array, line_indices=[0, 2, 5], labels=file_names)
#
# # Show the plot
# plt.show()