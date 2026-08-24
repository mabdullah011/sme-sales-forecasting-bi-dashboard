from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def moving_average_forecast(monthly_sales: pd.DataFrame, window: int = 3) -> pd.Series:
    series = monthly_sales["monthly_sales"].astype(float)
    return series.rolling(window=window, min_periods=1).mean()


def linear_regression_forecast(monthly_sales: pd.DataFrame) -> pd.Series:
    y = monthly_sales["monthly_sales"].astype(float).to_numpy()
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return pd.Series(model.predict(x), index=monthly_sales.index, name="lr_forecast")


def xgboost_forecast(monthly_sales: pd.DataFrame) -> pd.Series:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost to use xgboost_forecast.") from exc

    y = monthly_sales["monthly_sales"].astype(float).to_numpy()
    x = np.arange(len(y)).reshape(-1, 1)
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(x, y)
    preds = model.predict(x)
    return pd.Series(preds, index=monthly_sales.index, name="xgb_forecast")


def build_baseline_forecasts(monthly_sales: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "moving_average": moving_average_forecast(monthly_sales),
        "linear_regression": linear_regression_forecast(monthly_sales),
    }
