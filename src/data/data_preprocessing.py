import os
import logging
from typing import Any
import numpy as np
import pandas as pd

# Configure logging for the preprocessing module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log/data_preprocessing.log"),
        logging.StreamHandler()
    ]
)

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.

    Raises:
        Exception: If loading fails.
    """
    try:
        data = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def remove_duplicates(data: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.
        name (str, optional): Name for logging context.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    before = data.shape[0]
    data = data.drop_duplicates()
    after = data.shape[0]
    logging.info(f"Removed {before - after} duplicate rows from {name} data")
    return data

def save_data(data: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        path (str): Destination file path.

    Raises:
        Exception: If saving fails.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Failed to save data to {path}: {e}")
        raise

def preprocess(train_path: str, test_path: str, out_train_path: str, out_test_path: str) -> None:
    """
    Complete preprocessing pipeline: load, clean, and save data.

    Args:
        train_path (str): Path to raw train data.
        test_path (str): Path to raw test data.
        out_train_path (str): Path to save processed train data.
        out_test_path (str): Path to save processed test data.
    """
    try:
        train_data = load_data(train_path)
        test_data = load_data(test_path)

        train_data = remove_duplicates(train_data, "train")
        test_data = remove_duplicates(test_data, "test")

        save_data(train_data, out_train_path)
        save_data(test_data, out_test_path)

        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.critical(f"Data preprocessing failed: {e}")
        raise

def main() -> None:
    """
    Main entry point for data preprocessing.
    """
    preprocess(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv",
        out_train_path="data/processed/train.csv",
        out_test_path="data/processed/test.csv"
    )

if __name__ == "__main__":
    main()