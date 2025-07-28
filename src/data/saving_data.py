import pandas as pd

def save_dataframe(df: pd.DataFrame, path: str):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): File path where the CSV will be saved.
    """

    df.to_csv(path, index=False, encoding='utf-8')
    print(f"✅ Saved the dataframe at: {path}")