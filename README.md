# Time Series Sales Forecasting — Superstore Data

Forecasting a retail superstore's monthly sales using three classical time series
models — **AR**, **ARIMA**, and **SARIMA** — and comparing their forecast accuracy
on held-out data.

## Problem Statement

A company has historical sales data and wants to forecast future sales so it can
plan inventory more effectively.

> Given historical sales data of a company, forecast its future sales.

## Dataset

Order-level sales and profit data for 2011–2014, aggregated to **monthly** totals
for modeling.

| Attribute | Description |
|---|---|
| `Order Date` | The date the order was placed |
| `Sales` | Total sales value of the transaction (in dollars) |
| `Profit` | Profit made on the transaction (in dollars) |

## Workflow

1. **Data preparation** — parse dates, sort chronologically, resample to monthly sales
2. **Stationarity check** — Augmented Dickey-Fuller (ADF) test
3. **Transformations** — Box-Cox (stabilize variance) + differencing (stabilize mean)
4. **Order selection** — grid search `(p, q)` combinations, compare AIC/BIC
5. **Modeling** — fit AR, ARIMA, and SARIMA; forecast the held-out test period
6. **Evaluation** — compare models with RMSE and MAPE

### Train/Test Split

42 months of training data, remaining months held out for testing.

![Train/test split](images/train_test_split.png)

### Making the Series Stationary

The raw monthly sales series fails the ADF test (p ≈ 0.199), so it's transformed
with a log (Box-Cox, λ=0) transform followed by first-order differencing, which
passes the ADF test (p ≈ 7.45e-06).

![Box-Cox and differencing](images/boxcox_and_differencing.png)

Seasonal decomposition confirms a strong recurring yearly pattern in the original
series, which the transformation flattens out:

![Seasonal decomposition](images/seasonal_decomposition.png)

## Results

| Model | RMSE |
|---|---|
| AR | 15,094.49 |
| ARIMA | 24,353.77 |
| **SARIMA** | **11,178.61** |

![Model comparison](images/model_comparison_rmse.png)

**SARIMA performed best.** It's the only one of the three models that explicitly
models seasonality, and the data has a clear yearly cycle — so it tracks the
test-period peaks and troughs far more closely than plain AR or non-seasonal ARIMA.

| AR Forecast | ARIMA Forecast | SARIMA Forecast |
|---|---|---|
| ![AR forecast](../images/ar_model_forecast.png) | ![ARIMA forecast](images/arima_model_forecast.png) | ![SARIMA forecast](images/sarima_model_forecast.png) |

## Tech Stack

- Python, pandas, NumPy
- statsmodels (ADF test, seasonal decomposition, AR/ARIMA/SARIMA)
- scikit-learn (evaluation metrics)
- Matplotlib, Seaborn (visualization)

## Project Structure

```
.
├── time_series_sales_forecasting.ipynb   # main notebook
├── images/                               # saved plots used in this README
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Clone the repo and place `Superstore_Data.csv` in the project root
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `time_series_sales_forecasting.ipynb` and run all cells top-to-bottom

## Next Steps

- Re-run the AIC/BIC grid search directly on the transformed (Box-Cox + differenced)
  series rather than the raw series, for a fully consistent order selection
- Compare against the MAPE metric alongside RMSE for a fuller picture of accuracy
- Try `auto_arima` (pmdarima) or a Prophet baseline for comparison
