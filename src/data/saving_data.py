import os
import glob
from datetime import datetime
import pandas as pd


def _cleanup_old_versions(base_filename: str, directory: str, max_versions: int):
    """
    Remove old files that match the base_filename pattern beyond the latest `max_versions`.

    Parameters:
        base_filename (str): The base filename prefix to match.
        directory (str): The directory to look in.
        max_versions (int): The number of recent files to keep.
    """
    pattern = os.path.join(directory, f"{base_filename}_*.csv")
    files = sorted(glob.glob(pattern), reverse=True)
    
    # Remove older files beyond max_versions
    for old_file in files[max_versions:]:
        try:
            os.remove(old_file)
            print(f"🗑️ Deleted old file: {old_file}")
        except Exception as e:
            print(f"⚠️ Error deleting {old_file}: {e}")


def save_dataset(df: pd.DataFrame,
                 base_filename: str,
                 max_versions: int = 5,
                 directory: str = "../../data/processed") -> str:
    """
    Save a DataFrame to the specified directory with a timestamped filename.
    Keeps only the latest `max_versions` files for each base_filename.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        base_filename (str): The base name for the file (e.g., "resumes_cleaned").
        max_versions (int): How many recent files to keep.
        directory (str): Target directory to save the files.

    Returns:
        str: The full path of the saved file.
    """

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.csv"
    filepath = os.path.join(directory, filename)

    # Save the DataFrame
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"✅ Saved: {filepath}")

    # Clean up old versions
    _cleanup_old_versions(base_filename, directory, max_versions)

    return filepath