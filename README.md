# SME Sales Forecasting & Business Intelligence Dashboard

An end-to-end Python analytics starter project for small and medium businesses to analyze historical sales, discover customer and product insights, and forecast future sales.

## Business Problem
SMEs often have fragmented sales data and limited analytics capability, making it difficult to anticipate demand, optimize inventory, and identify profitable customer/product segments.

## Project Objectives
- Build a reproducible sales analytics workflow from raw data to business insights.
- Perform data quality validation and cleaning for reliable reporting.
- Explore sales patterns by time, category, product, customer, and region.
- Create baseline forecasting models for short-term sales planning.
- Provide business-oriented recommendations for growth and efficiency.

## Dataset
The project expects retail transaction data with these columns:
- `order_id`, `order_date`, `customer_id`, `product_name`, `category`, `quantity`, `unit_price`, `sales`, `region`, `payment_method`

If no dataset is found at `data/raw/retail_sales.csv`, a realistic synthetic sample dataset is automatically generated.

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn, statsmodels
- Optional: XGBoost
- Jupyter Notebooks

## Project Structure
```text
data/
  raw/
  processed/

notebooks/
  01_data_cleaning.ipynb
  02_exploratory_data_analysis.ipynb
  03_sales_forecasting.ipynb

src/
  data_preprocessing.py
  feature_engineering.py
  forecasting_model.py
  evaluation.py

dashboard/
  dashboard_requirements.md

reports/
  business_insights.md

tests/
  test_preprocessing.py

README.md
requirements.txt
.gitignore
LICENSE
```

## Workflow
1. **Data Loading & Generation**: load local data or auto-generate sample data.
2. **Data Quality Checks**: missing values, duplicates, invalid dates, negative values, and outliers.
3. **Data Cleaning**: remove invalid rows and persist cleaned output.
4. **EDA**: monthly trend, category performance, regional sales, top products, and customer behavior.
5. **Forecasting**:
   - Moving average baseline
   - Linear regression baseline
   - Optional XGBoost model
6. **Evaluation**: MAE, RMSE, and MAPE.
7. **Business Reporting**: recommendations in `reports/business_insights.md`.

## Results (Placeholders)
- **Data Quality**: _Add summary of cleaning impact (rows removed, missing fixed, etc.)_
- **EDA**: _Add key charts and insights_
- **Forecast Performance**: _Add MAE/RMSE/MAPE table by model_
- **Business Recommendations**: _Add prioritized actions and expected impact_

## Installation & Local Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Run preprocessing:
```bash
python src/data_preprocessing.py
```

Run tests:
```bash
pytest tests/test_preprocessing.py
```

Open notebooks:
```bash
jupyter notebook
```

## Future Improvements
- Add time-series models (ARIMA/SARIMA/Prophet) and model selection.
- Add feature store and experiment tracking.
- Build an interactive dashboard in Streamlit/Power BI.
- Automate ETL and reporting with scheduled pipelines.
- Add anomaly detection for sales and inventory alerts.

## License
This project is open-source and available under the MIT License.
