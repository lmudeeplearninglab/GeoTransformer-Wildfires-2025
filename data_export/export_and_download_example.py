# coding=utf-8
"""Script for exporting wildfire data and downloading to local exports folder.

This script works similarly to export_dataset.ipynb:
1. Authenticates and initializes Earth Engine
2. Configures export parameters
3. Triggers TFRecord exports using data_export.export_ee_data
4. Monitors export tasks
5. Downloads exported files from GCS to local exports folder
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Handle imports when running directly vs as a module
try:
    from data_export import export_ee_data, ee_utils
except ImportError:
    # If running directly, add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from data_export import export_ee_data, ee_utils

import ee


def authenticate_ee():
    """Authenticate with Earth Engine."""
    try:
        ee.Authenticate()
    except Exception as exc:
        print("Auth skipped or already configured:", exc)


def initialize_ee():
    """Initialize Earth Engine."""
    ee.Initialize()
    print("Earth Engine initialized", datetime.utcnow())


def export_slice(config: dict):
    """Export data slice using configuration dictionary.
    
    This function works similarly to the notebook's export_slice function.
    
    Args:
        config: Dictionary with export parameters:
            - bucket: GCS bucket name
            - folder: Folder in bucket
            - prefix: File prefix
            - start_date: Start date (YYYY-MM-DD)
            - end_date: End date (YYYY-MM-DD)
            - kernel_size: Size of exported tiles (default: 64)
            - sampling_scale: Resolution in meters (default: 1000)
            - num_samples_per_file: Samples per file (default: 100)
            - region_bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
    """
    required = ["bucket", "folder", "prefix", "start_date", "end_date"]
    for key in required:
        if not config.get(key):
            raise ValueError(f"Missing required parameter: {key}")

    bbox = config.get("region_bbox") or ee_utils.COORDINATES["US"]
    # Override the default region used by export_ee_data
    ee_utils.COORDINATES["US"] = bbox

    start_date = ee.Date(config["start_date"])
    end_date = ee.Date(config["end_date"])
    
    # Convert bbox to ee.Geometry.Rectangle
    geometry = ee.Geometry.Rectangle(bbox)

    export_ee_data.export_single_fire_dataset(
        bucket=config["bucket"],
        folder=config["folder"],
        start_date=start_date,
        end_date=end_date,
        geometry=geometry,
        prefix=config.get("prefix", ""),
        kernel_size=config.get("kernel_size", ee_utils.DEFAULT_KERNEL_SIZE),
        sampling_scale=config.get("sampling_scale", ee_utils.DEFAULT_SAMPLING_RESOLUTION),
        num_samples_per_file=config.get("num_samples_per_file", ee_utils.DEFAULT_LIMIT_PER_EE_CALL),
    )
    print("Export triggered. Check https://code.earthengine.google.com/tasks for progress.")


def list_tasks(limit: int = 10, prefix_filter: str = None):
    """List Earth Engine export tasks.
    
    Args:
        limit: Maximum number of tasks to list
        prefix_filter: Optional prefix to filter tasks by description
    """
    tasks = ee.batch.Task.list()
    filtered_tasks = tasks
    if prefix_filter:
        filtered_tasks = [t for t in tasks if prefix_filter in t.config.get('description', '')]
    
    for task in filtered_tasks[:limit]:
        status = task.status()
        print(f"{status['id']} | {status.get('state')} | {status.get('description')}")
    
    return filtered_tasks[:limit]


def wait_for_tasks(prefix_filter: str, timeout: int = 3600):
    """Wait for Earth Engine export tasks to complete.
    
    Args:
        prefix_filter: Prefix to filter tasks by description
        timeout: Maximum time to wait in seconds (default: 1 hour)
    
    Returns:
        True if all tasks completed, False if timeout
    """
    import time
    start_time = time.time()
    
    print(f"Waiting for tasks with prefix '{prefix_filter}' to complete...")
    print("(This may take several minutes to hours depending on data size)")
    
    while True:
        tasks = ee.batch.Task.list()
        filtered_tasks = [t for t in tasks if prefix_filter in t.config.get('description', '')]
        
        if not filtered_tasks:
            print("No tasks found with the specified prefix.")
            return True
        
        # Check task statuses
        all_complete = True
        running_tasks = []
        for task in filtered_tasks:
            status = task.status()
            state = status.get('state', 'UNKNOWN')
            if state in ['READY', 'RUNNING']:
                all_complete = False
                running_tasks.append(status.get('description', 'Unknown'))
            elif state == 'FAILED':
                print(f"WARNING: Task {status.get('description')} FAILED!")
                print(f"  Error: {status.get('error_message', 'Unknown error')}")
        
        if all_complete:
            print("All tasks completed!")
            return True
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"Timeout after {timeout} seconds. Some tasks may still be running.")
            return False
        
        if running_tasks:
            print(f"  Still running: {len(running_tasks)} tasks (elapsed: {int(elapsed)}s)")
        
        time.sleep(30)  # Check every 30 seconds


def download_exports(config: dict, destination: Path):
    """Download exported TFRecords from GCS bucket.
    
    This function works similarly to the notebook's download_exports function.
    It attempts to use Earth Engine credentials or Application Default Credentials.
    
    Args:
        config: Dictionary with download parameters:
            - bucket: GCS bucket name
            - folder: Folder in bucket
            - prefix: Optional file prefix to filter downloads
        destination: Local directory path to download files to
    """
    return ee_utils.download_exports_from_gcs(
        bucket_name=config["bucket"],
        folder=config.get("folder", ""),
        destination=destination,
        prefix=config.get("prefix"),
    )


def main(wait_for_completion: bool = False, auto_download: bool = True, exports_dir: Path = None):
    """Main function demonstrating the export and download workflow.
    
    Args:
        wait_for_completion: If True, wait for exports to complete before downloading
        auto_download: If True, automatically download files after exports complete
        exports_dir: Directory to download files to (default: project exports folder)
    """
    # Set default exports directory to project exports folder
    if exports_dir is None:
        # Get the project root (parent of data_export folder)
        project_root = Path(__file__).parent.parent
        exports_dir = project_root / "exports"
    
    # Step 1: Authenticate and initialize
    print("Step 1: Authenticating with Earth Engine...")
    authenticate_ee()
    initialize_ee()
    
    # Step 2: Configure export parameters
    print("\nStep 2: Configuring export parameters...")
    params = {
        "bucket": "lmudl-wildfire-compilation-bucket",  # Change to your bucket
        "folder": "palisades",
        "prefix": "palisades_sample",
        "start_date": "2025-01-01",
        "end_date": "2025-02-05",
        "kernel_size": 64,
        "sampling_scale": 1000,
        "eval_split_ratio": 0.05,
        "num_samples_per_file": 100,
        "region_bbox": [-118.686534, 34.129427, 
                        -118.50074, 34.030351]
    }
    print(json.dumps(params, indent=2))
    print(f"\nFiles will be downloaded to: {exports_dir.resolve()}")
    
    # Step 3: Export data
    print("\nStep 3: Exporting data to GCS...")
    export_slice(params)
    
    # Step 4: List tasks
    print("\nStep 4: Listing export tasks...")
    list_tasks(limit=10, prefix_filter=params["prefix"])
    
    # Step 5: Wait for completion (if requested)
    if wait_for_completion:
        print("\nStep 5: Waiting for exports to complete...")
        all_complete = wait_for_tasks(params["prefix"])
        if not all_complete:
            print("\nExports are still running. You can download files later.")
            print("Run this script again with --download-only to download completed files.")
            return
    else:
        print("\nStep 5: Exports are running in the background.")
        print("Check task status at: https://code.earthengine.google.com/tasks")
        print("\nTo download files after exports complete, run:")
        print(f"  python {__file__} --download-only")
        if not auto_download:
            return
    
    # Step 6: Download files
    if auto_download:
        print("\nStep 6: Downloading files from GCS...")
        try:
            downloaded = download_exports(params, exports_dir)
            if downloaded:
                print(f"\n✓ Successfully downloaded {len(downloaded)} files to {exports_dir.resolve()}!")
                for f in downloaded:
                    print(f"  - {f}")
            else:
                print("\n⚠ No files were downloaded. This could mean:")
                print("  1. Exports are not yet complete")
                print("  2. No files match the prefix/folder")
                print("  3. Check task status at: https://code.earthengine.google.com/tasks")
        except Exception as e:
            print(f"\n✗ Error downloading files: {e}")
            print("\nMake sure you have:")
            print("1. Authenticated with gcloud: gcloud auth application-default login")
            print("2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("3. And that exports have completed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export wildfire data and download to local exports folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export only (don't wait or download)
  python export_and_download_example.py
  
  # Export and wait for completion, then download
  python export_and_download_example.py --wait --download
  
  # Download only (after exports are complete)
  python export_and_download_example.py --download-only
        """
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Wait for exports to complete before downloading'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        default=True,
        help='Download files after export (default: True)'
    )
    parser.add_argument(
        '--no-download',
        dest='download',
        action='store_false',
        help='Skip downloading files'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download files (skip export)'
    )
    parser.add_argument(
        '--exports-dir',
        type=str,
        default=None,
        help='Directory to download files to (default: project exports folder)'
    )
    
    args = parser.parse_args()
    
    # Set exports directory
    if args.exports_dir:
        exports_dir = Path(args.exports_dir)
    else:
        # Default to project exports folder
        project_root = Path(__file__).parent.parent
        exports_dir = project_root / "exports"
    
    if args.download_only:
        # Download only mode
        print("Download-only mode: Skipping export...")
        authenticate_ee()
        initialize_ee()
        
        params = {
            "bucket": "lmudl-wildfire-compilation-bucket",
            "folder": "eaton",
            "prefix": "eaton_sample",
        }
        
        print(f"\nDownloading files from GCS to: {exports_dir.resolve()}")
        try:
            downloaded = download_exports(params, exports_dir)
            if downloaded:
                print(f"\n✓ Successfully downloaded {len(downloaded)} files to {exports_dir.resolve()}!")
                for f in downloaded:
                    print(f"  - {f}")
            else:
                print("\n⚠ No files were downloaded.")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        # Normal export mode
        main(wait_for_completion=args.wait, auto_download=args.download, exports_dir=exports_dir)

