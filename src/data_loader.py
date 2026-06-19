import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load Superstore dataset.
    """

    df = pd.read_csv(
        file_path,
        encoding="latin1"
    )

    return df