import json
import pandas as pd

from .core import MethodConfig

def load_measurements(filepath: str) -> pd.DataFrame:
    """
    Load the 2x2 scouting measurements from a CSV file.
    """
    measurements = pd.read_csv(filepath, index_col=False, skipinitialspace=True)
    measurements = measurements.set_index("analyte")
    return measurements

def load_config(filepath: str) -> MethodConfig:
    """
    Load the method/design configuration from a JSON file.
    """
    with open(filepath, "r") as f:
        config_dict = json.load(f)
    return MethodConfig(**config_dict)