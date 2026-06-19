import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.
    """

    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    """

    df["row_id"] = df["row_id"].astype("object")
    df["postal_code"] = df["postal_code"].astype("object")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features.
    """

    df["shipping_days"] = (
        df["ship_date"] - df["order_date"]
    ).dt.days

    return df


def prepare_features_target(df: pd.DataFrame):
    """
    Prepare model features and target.
    """

    X = df[
        [
            "sales",
            "discount",
            "quantity",
            "segment",
            "category",
            "sub_category",
            "region",
            "ship_mode"
        ]
    ]

    y = df["profit"]

    X = pd.get_dummies(
        X,
        columns=[
            "segment",
            "category",
            "sub_category",
            "region",
            "ship_mode"
        ],
        drop_first=True
    )

    return X, y