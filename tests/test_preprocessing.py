from pathlib import Path

import pandas as pd

from src.data_preprocessing import (
    clean_sales_data,
    ensure_dataset_exists,
    load_sales_data,
    run_data_quality_checks,
)


def test_ensure_dataset_exists_generates_sample_data(tmp_path):
    data_path = tmp_path / "retail_sales.csv"

    resolved = ensure_dataset_exists(data_path)

    assert resolved.exists()
    df = pd.read_csv(resolved)
    assert set(
        [
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
    ).issubset(df.columns)


def test_clean_sales_data_removes_invalid_rows(tmp_path):
    csv_path = tmp_path / "input.csv"
    raw = pd.DataFrame(
        {
            "order_id": ["1", "1", "2"],
            "order_date": ["2024-01-01", "2024-01-01", "invalid-date"],
            "customer_id": ["c1", "c1", "c2"],
            "product_name": ["A", "A", "B"],
            "category": ["Cat", "Cat", "Cat"],
            "quantity": [2, 2, -1],
            "unit_price": [10.0, 10.0, 20.0],
            "sales": [20.0, 20.0, -20.0],
            "region": ["North", "North", "South"],
            "payment_method": ["Card", "Card", "Cash"],
        }
    )
    raw.to_csv(csv_path, index=False)

    loaded = load_sales_data(csv_path)
    cleaned, report = clean_sales_data(loaded)

    assert report.duplicate_rows == 1
    assert report.invalid_dates == 1
    assert report.negative_quantities == 1
    assert len(cleaned) == 1


def test_quality_report_contains_expected_keys(tmp_path):
    data_path = tmp_path / "retail_sales.csv"
    df = load_sales_data(data_path)

    report = run_data_quality_checks(df)

    assert "sales" in report.missing_values
    assert report.negative_prices >= 0
