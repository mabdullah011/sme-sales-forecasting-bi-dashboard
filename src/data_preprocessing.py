from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "sales",
    "region",
    "payment_method",
]


@dataclass
class DataQualityReport:
    missing_values: Dict[str, int]
    duplicate_rows: int
    invalid_dates: int
    negative_quantities: int
    negative_prices: int
    quantity_outliers: int
    unit_price_outliers: int


def _create_sample_dataset(rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    categories = {
        "Electronics": ["Tablet", "Laptop", "Smartphone", "Headphones"],
        "Office Supplies": ["Printer Paper", "Desk Chair", "Notebook", "Pen Set"],
        "Home": ["Blender", "Air Fryer", "Vacuum", "Lamp"],
        "Fashion": ["Jacket", "Sneakers", "Backpack", "Watch"],
    }
    regions = ["North", "South", "East", "West"]
    payment_methods = ["Cash", "Card", "Bank Transfer", "Mobile Wallet"]

    category_choices = rng.choice(list(categories.keys()), size=rows)
    product_names = [rng.choice(categories[cat]) for cat in category_choices]
    quantities = rng.integers(1, 8, size=rows)
    unit_prices = np.round(rng.uniform(5, 350, size=rows), 2)

    order_dates = pd.date_range("2023-01-01", periods=730, freq="D")
    sampled_dates = rng.choice(order_dates, size=rows, replace=True)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + i}" for i in range(rows)],
            "order_date": pd.to_datetime(sampled_dates),
            "customer_id": [f"CUST-{rng.integers(1000, 5000)}" for _ in range(rows)],
            "product_name": product_names,
            "category": category_choices,
            "quantity": quantities,
            "unit_price": unit_prices,
            "region": rng.choice(regions, size=rows),
            "payment_method": rng.choice(payment_methods, size=rows),
        }
    )
    df["sales"] = np.round(df["quantity"] * df["unit_price"], 2)

    return df.sort_values("order_date").reset_index(drop=True)


def ensure_dataset_exists(raw_data_path: str | Path) -> Path:
    """Generate a realistic sample dataset when source data is unavailable."""
    raw_data_path = Path(raw_data_path)
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_data_path.exists():
        _create_sample_dataset().to_csv(raw_data_path, index=False)
    return raw_data_path


def load_sales_data(raw_data_path: str | Path) -> pd.DataFrame:
    path = ensure_dataset_exists(raw_data_path)
    df = pd.read_csv(path)

    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


def _outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def run_data_quality_checks(df: pd.DataFrame) -> DataQualityReport:
    return DataQualityReport(
        missing_values=df.isna().sum().to_dict(),
        duplicate_rows=int(df.duplicated().sum()),
        invalid_dates=int(df["order_date"].isna().sum()),
        negative_quantities=int((df["quantity"] < 0).sum()),
        negative_prices=int((df["unit_price"] < 0).sum()),
        quantity_outliers=_outlier_count(df["quantity"]),
        unit_price_outliers=_outlier_count(df["unit_price"]),
    )


def clean_sales_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    report_before = run_data_quality_checks(df)

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=["order_date"])
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)]
    cleaned["sales"] = cleaned["quantity"] * cleaned["unit_price"]

    return cleaned.reset_index(drop=True), report_before


def save_processed_data(df: pd.DataFrame, processed_path: str | Path) -> Path:
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return processed_path


if __name__ == "__main__":
    raw_file = Path("data/raw/retail_sales.csv")
    processed_file = Path("data/processed/retail_sales_cleaned.csv")

    data = load_sales_data(raw_file)
    cleaned_data, quality_report = clean_sales_data(data)
    save_processed_data(cleaned_data, processed_file)

    print("Dataset ready:", raw_file)
    print("Processed dataset:", processed_file)
    print("Data quality report:", quality_report)
