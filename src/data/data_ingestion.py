import os
import logging
from typing import Tuple, Dict, Any
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_ingestion.log"),
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> Dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise

def fetch_data(url: str) -> pd.DataFrame:
    """Fetch data from a CSV URL or file path."""
    try:
        data = pd.read_csv(url)
        logging.info(f"Data loaded from {url} with shape {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {url}: {e}")
        raise

def split_data(
    data: pd.DataFrame, 
    test_size: float, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    try:
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train ({train.shape}) and test ({test.shape})")
        return train, test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(data: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        test_size = params["data_ingestion"]["test_size"]
        url = params["data_ingestion"]["url"]

        data = fetch_data(url)
        train_data, test_data = split_data(data, test_size)

        save_data(train_data, "data/raw/train.csv")
        save_data(test_data, "data/raw/test.csv")

        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.critical(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()