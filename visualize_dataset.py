# coding=utf-8
"""Script for loading and visualizing wildfire datasets.

This script works similarly to dataset_demo.ipynb:
1. Loads TFRecord datasets using dataset.py
2. Filters features by availability
3. Visualizes the data with matplotlib
"""

import sys
from pathlib import Path

# Handle imports when running directly vs as a module
try:
    import constants
    import dataset
except ImportError:
    # If running directly, add parent directory to path
    parent_dir = Path(__file__).parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import constants
    import dataset

import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf


def load_and_visualize_dataset(
    file_pattern: str = None,
    data_size: int = 64,
    sample_size: int = 32,
    output_sample_size: int = 32,
    batch_size: int = 100,
    n_rows: int = 5,
    save_plot: bool = False,
    output_path: str = None,
):
    """Load and visualize a wildfire dataset.
    
    Args:
        file_pattern: Glob pattern for TFRecord files (default: project exports folder)
        data_size: Size of tiles in pixels (square) as read from input files
        sample_size: Size of input samples after processing
        output_sample_size: Size of output samples after processing
        batch_size: Batch size for dataset
        n_rows: Number of rows to visualize
        save_plot: Whether to save the plot to a file
        output_path: Path to save the plot if save_plot is True (default: project root)
    """
    # Set default file pattern to project exports folder
    if file_pattern is None:
        project_root = Path(__file__).parent
        exports_dir = project_root / "exports"
        file_pattern = str(exports_dir / "palisades_sample*")
    
    # Set default output path
    if output_path is None:
        project_root = Path(__file__).parent
        output_path = str(project_root / "dataset_visualization.png")
    
    print(f"Loading dataset from: {file_pattern}")
    
    # Automatically detect and filter features available in the TFRecords
    print("\nFiltering features by availability...")
    INPUT_FEATURES = dataset.filter_features_by_availability(
        constants.INPUT_FEATURES, 
        file_pattern,
        verbose=True
    )
    
    print(f"\nUsing {len(INPUT_FEATURES)} input features: {INPUT_FEATURES}")
    
    # Load the dataset
    print("\nLoading dataset...")
    train_dataset = dataset.get_dataset(
        file_pattern=file_pattern,
        data_size=data_size,
        sample_size=sample_size,
        output_sample_size=output_sample_size,
        batch_size=batch_size,
        input_features=INPUT_FEATURES,
        output_features=constants.OUTPUT_FEATURES,
        shuffle=False,
        shuffle_buffer_size=1000,
        compression_type='GZIP',
        input_sequence_length=1,
        output_sequence_length=1,
        repeat=False,
        clip_and_normalize=False,
        clip_and_rescale=False,
        random_flip=False,
        random_rotate=False,
        random_crop=False,
        center_crop=True,
        azimuth_in_channel=None,
        azimuth_out_channel=None,
        downsample_threshold=0.0,
        binarize_output=True
    )
    
    # Materialize the first batch
    print("\nMaterializing first batch...")
    inputs, labels = next(iter(train_dataset))
    print(f"Input shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create visualization (matching dataset_demo.ipynb)
    print("\nCreating visualization...")
    n_features = inputs.shape[3]
    
    # TITLES list matching the notebook (in order of all possible features)
    TITLES = [
        'Elevation',
        'Drought',
        'Vegetation',
        'Precip',
        'Humidity',
        'Wind\ndirection',
        'Min\ntemp',
        'Max\ntemp',
        'Wind\nvelocity',
        'Energy\nrelease\ncomponent',
        'Previous\nfire\nmask',
        'Fire\nmask'
    ]
    
    # Create colormap for fire masks (matching notebook)
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    keys = INPUT_FEATURES
    
    # Create figure (matching notebook style)
    fig = plt.figure(figsize=(15, 6.5 * (n_rows / 5)))
    
    for i in range(n_rows):
        for j in range(n_features + 1):
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            if i == 0:
                plt.title(TITLES[j], fontsize=13)
            if j < n_features - 1:
                # All input features except the last one use viridis
                plt.imshow(inputs[i, :, :, j], cmap='viridis')
            if j == n_features - 1:
                # Last input feature uses inferno (matching notebook)
                im = plt.imshow(inputs[i, :, :, -1], cmap='inferno', vmin=0, vmax=9)
            if j == n_features:
                # Output feature (FireMask) uses inferno
                im = plt.imshow(labels[i, :, :, 0], cmap='inferno', vmin=0, vmax=9)
            plt.axis('off')
    
    plt.tight_layout()
    
    # Add colorbar for the inferno colormap (fire mask)
    cbar = fig.colorbar(im, ax=fig.axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Fire Intensity', rotation=270, labelpad=15)
    
    if save_plot:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()
    
    return inputs, labels, INPUT_FEATURES


def main():
    """Main function for dataset visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize wildfire dataset')
    parser.add_argument(
        '--file_pattern',
        type=str,
        default=None,
        help='Glob pattern for TFRecord files (default: project exports folder/eaton_sample*)'
    )
    parser.add_argument(
        '--data_size',
        type=int,
        default=64,
        help='Size of tiles in pixels (default: 64)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=32,
        help='Size of input samples after processing (default: 32)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size (default: 100)'
    )
    parser.add_argument(
        '--n_rows',
        type=int,
        default=23,
        help='Number of rows to visualize (default: 23, matching notebook)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save plot to file instead of displaying'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset_visualization.png',
        help='Output path for saved plot (default: dataset_visualization.png)'
    )
    
    args = parser.parse_args()
    
    load_and_visualize_dataset(
        file_pattern=args.file_pattern,
        data_size=args.data_size,
        sample_size=args.sample_size,
        output_sample_size=args.sample_size,
        batch_size=args.batch_size,
        n_rows=args.n_rows,
        save_plot=args.save,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()

