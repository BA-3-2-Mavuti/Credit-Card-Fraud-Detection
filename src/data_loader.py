from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

REQUIRED_COLUMNS = [f'V{i}' for i in range(1,29)] + ['Amount','Class']

def load_creditcard_csv(path: str = None) -> pd.DataFrame:
    """
    Loads the CSV file, checks for required columns, and inserts a synthetic Time column if missing.
    
    Parameters:
        path: Path to CSV file. Defaults to './data/creditcard.csv' if None.
        
    Returns:
        pd.DataFrame: Loaded and validated dataset.
    """
    path = Path(path) if path else Path.cwd() / "data" / "creditcard.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logging.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    
    # Add Time column if missing
    if 'Time' not in df.columns:
        logging.warning("'Time' column missing – adding synthetic Time values.")
        df.insert(1, 'Time', np.arange(len(df)))

    # Checks for missing required columns (excluding Time)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    logging.info("Dataset loaded successfully.")
    return df