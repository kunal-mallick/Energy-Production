import os
import yaml
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging for the feature engineering module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)

def load_config(config_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.

    Raises:
        Exception: If loading fails.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from CSV files.

    Args:
        train_path (str): Path to train data CSV.
        test_path (str): Path to test data CSV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.

    Raises:
        Exception: If loading fails.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Train data loaded from {train_path} with shape {train_data.shape}")
        logging.info(f"Test data loaded from {test_path} with shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load train/test data: {e}")
        raise

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    try:
        df = df.copy()
        df["temp_humidity_interaction"] = df["temperature"] / (df["r_humidity"] + 1)
        df["pressure_temperature_product"] = df["amb_pressure"] * df["temperature"]
        df["vacuum_efficiency_factor"] = df["exhaust_vacuum"] / (df["amb_pressure"] + 1)
        df["saturation_indicator"] = df["r_humidity"] * df["temperature"]
        logging.info("New features created successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to create features: {e}")
        raise

def scale_features(xtrain: pd.DataFrame, xtest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale features using StandardScaler.

    Args:
        xtrain (pd.DataFrame): Training features.
        xtest (pd.DataFrame): Testing features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled train and test features.
    """
    try:
        scaler = StandardScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        logging.info("Features scaled successfully.")
        return xtrain_scaled, xtest_scaled
    except Exception as e:
        logging.error(f"Failed to scale features: {e}")
        raise

def apply_pca(xtrain: np.ndarray, xtest: np.ndarray, n_components: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA for dimensionality reduction.

    Args:
        xtrain (np.ndarray): Scaled training features.
        xtest (np.ndarray): Scaled testing features.
        n_components (int): Number of PCA components.
        random_state (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: PCA-transformed train and test features.
    """
    try:
        pca = PCA(n_components=n_components, random_state=random_state)
        xtrain_pca = pca.fit_transform(xtrain)
        xtest_pca = pca.transform(xtest)
        logging.info("PCA applied successfully.")
        return xtrain_pca, xtest_pca
    except Exception as e:
        logging.error(f"Failed to apply PCA: {e}")
        raise

def save_features(x: np.ndarray, y: pd.Series, path: str) -> None:
    """
    Save features and target to a CSV file.

    Args:
        x (np.ndarray): Feature array.
        y (pd.Series): Target variable.
        path (str): Destination file path.

    Raises:
        Exception: If saving fails.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(x)
        df['energy_production'] = y.values
        df.to_csv(path, index=False)
        logging.info(f"Features saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save features to {path}: {e}")
        raise

def main() -> None:
    """
    Main entry point for feature engineering pipeline.
    Loads config, data, creates features, scales, applies PCA, and saves results.
    """
    try:
        config = load_config("params.yaml")
        n_components = config["feature_engineering"]["n_components"]
        random_state = config["feature_engineering"]["random_state"]

        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")

        # Split features and target
        xtrain = train_data.drop(columns=["energy_production"])
        ytrain = train_data["energy_production"]
        xtest = test_data.drop(columns=["energy_production"])
        ytest = test_data["energy_production"]

        # Feature creation
        xtrain = create_features(xtrain)
        xtest = create_features(xtest)

        # Scaling
        xtrain_scaled, xtest_scaled = scale_features(xtrain, xtest)

        # PCA
        xtrain_pca, xtest_pca = apply_pca(xtrain_scaled, xtest_scaled, n_components, random_state)

        # Save processed features
        save_features(xtrain_pca, ytrain, "data/interim/train_pca.csv")
        save_features(xtest_pca, ytest, "data/interim/test_pca.csv")

        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()