from __future__ import annotations

import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["order_date"] = pd.to_datetime(featured["order_date"], errors="coerce")
    featured["year"] = featured["order_date"].dt.year
    featured["month"] = featured["order_date"].dt.month
    featured["quarter"] = featured["order_date"].dt.quarter
    featured["day_of_week"] = featured["order_date"].dt.dayofweek
    return featured


def aggregate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["order_date"] = pd.to_datetime(temp["order_date"], errors="coerce")
    monthly = (
        temp.dropna(subset=["order_date"])
        .set_index("order_date")
        .resample("MS")["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "monthly_sales"})
    )
    return monthly
