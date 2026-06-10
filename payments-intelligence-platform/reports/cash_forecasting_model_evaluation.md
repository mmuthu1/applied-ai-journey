# Cash Forecasting Model Evaluation Report

## 1. Executive Summary

This report evaluates the first cash forecasting models built for the Payments Intelligence Platform.

The objective is to forecast the next day’s total payment amount using daily payment activity, rolling averages, payment counts, failed payment rates, high-value payment rates, and time-based features.

The current best experimental forecasting model is:

```text
Random Forest Regressor
```

The Random Forest model slightly outperformed the strongest baseline, the 7-day moving average forecast.

Key findings:

* The cash forecasting dataset contains 173 daily records.
* A time-based train/test split was used to simulate real forecasting.
* The best baseline was the 7-day moving average forecast.
* Random Forest Regressor achieved the lowest MAE and MAPE.
* The improvement over baseline was modest but meaningful.
* More historical data and stronger time-series features are needed before production use.

## 2. Business Problem

Payment operations and cash management teams need forward-looking visibility into expected payment volumes and payment amounts.

A cash forecasting model can help answer:

```text
What is tomorrow’s expected total payment amount?
```

Better forecasts may support:

* Liquidity planning
* Operational staffing
* Exception monitoring
* Cash-position awareness
* Risk-based payment operations planning

The goal is not simply to train a model. The goal is to determine whether a machine learning forecast improves over simple business baselines.

## 3. Dataset Summary

The forecasting dataset is:

```text
data/processed/cash_forecast_daily.csv
```

The dataset was created by aggregating transaction-level payment data into daily records.

Dataset summary:

```text
Rows: 173
Columns: 22
Date range: 2024-01-07 to 2024-06-27
```

The target column is:

```text
next_day_total_amount
```

This means the model uses today’s daily activity and recent historical features to predict tomorrow’s total payment amount.

## 4. Forecasting Features

The forecasting dataset includes daily aggregate features such as:

| Feature                       | Description                                   |
| ----------------------------- | --------------------------------------------- |
| `daily_payment_count`         | Number of payments on the day                 |
| `daily_total_amount`          | Total payment amount on the day               |
| `daily_average_amount`        | Average payment amount on the day             |
| `daily_median_amount`         | Median payment amount on the day              |
| `failed_payment_count`        | Number of failed payments                     |
| `high_value_payment_count`    | Number of high-value payments                 |
| `failed_payment_rate`         | Failed payments divided by total payments     |
| `high_value_payment_rate`     | High-value payments divided by total payments |
| `previous_day_total_amount`   | Prior day total payment amount                |
| `previous_day_payment_count`  | Prior day payment count                       |
| `rolling_3_day_avg_amount`    | 3-day rolling average of total payment amount |
| `rolling_7_day_avg_amount`    | 7-day rolling average of total payment amount |
| `rolling_3_day_payment_count` | 3-day rolling average of payment count        |
| `rolling_7_day_payment_count` | 7-day rolling average of payment count        |
| `day_of_week`                 | Day of week                                   |
| `month`                       | Month                                         |
| `day_of_month`                | Day of month                                  |
| `is_weekend`                  | Weekend flag                                  |

## 5. Why Time-Based Train/Test Split Was Used

For forecasting, the model must predict future values using past data.

Therefore, a random train/test split is not appropriate because it can allow information from future dates to influence training.

This project uses a time-based split:

```text
Training period: 2024-01-07 to 2024-05-23
Test period:     2024-05-24 to 2024-06-27
```

This better simulates real forecasting behavior.

## 6. Baseline Forecasts

Before training machine learning models, simple baseline forecasts were evaluated.

The baseline methods were:

1. Previous day amount
2. 3-day moving average amount
3. 7-day moving average amount

The best baseline was:

```text
7-day moving average forecast
```

Baseline performance:

```text
MAE:  $923,898.74
RMSE: $1,204,937.67
MAPE: 32.27%
```

This baseline is important because a machine learning model should beat it to be useful.

## 7. Evaluation Metrics

The forecasting models were evaluated using:

| Metric | Meaning                                                 |
| ------ | ------------------------------------------------------- |
| MAE    | Mean Absolute Error; average dollar miss                |
| RMSE   | Root Mean Squared Error; penalizes large misses more    |
| MAPE   | Mean Absolute Percentage Error; average percentage miss |

For business interpretation, MAE is especially useful because it answers:

```text
On average, how many dollars is the forecast off by?
```

## 8. Models Evaluated

The following forecasting methods were compared:

1. 7-day moving average baseline
2. Linear Regression
3. Random Forest Regressor

Results:

| Model                         |         MAE |          RMSE |   MAPE | Improvement vs Baseline |
| ----------------------------- | ----------: | ------------: | -----: | ----------------------: |
| Random Forest Regressor       | $894,179.20 | $1,163,787.60 | 30.00% |                  +3.22% |
| 7-day moving average baseline | $923,898.74 | $1,204,937.67 | 32.27% |                Baseline |
| Linear Regression             | $926,054.91 | $1,134,513.00 | 31.59% |                  -0.23% |

## 9. Model Interpretation

The current best experimental model is:

```text
Random Forest Regressor
```

The Random Forest model improved MAE by:

```text
$29,719.55
```

or:

```text
3.22%
```

compared with the 7-day moving average baseline.

This improvement is modest, but it is meaningful because the model was evaluated against a reasonable baseline.

Linear Regression had a slightly lower RMSE than Random Forest, but Random Forest performed better on MAE and MAPE. Since MAE is easier to explain as average dollar error, Random Forest is the preferred experimental model at this stage.

## 10. Feature Importance

The top Random Forest feature importance signals were:

* `rolling_7_day_payment_count`
* `daily_average_amount`
* `daily_median_amount`
* `daily_payment_count`
* `previous_day_total_amount`
* `day_of_month`
* `rolling_3_day_avg_amount`
* `failed_payment_rate`
* `rolling_7_day_avg_amount`
* `daily_total_amount`
* `high_value_payment_rate`
* `day_of_week`

These signals make business sense because next-day payment amount is related to:

* Recent payment volume
* Recent payment amount trends
* Average and median payment size
* Calendar timing
* Failed payment and high-value payment behavior

## 11. Operational Recommendation

The current recommendation is:

```text
Use Random Forest Regressor as the current experimental forecasting model.
```

However, this model should not yet be considered production-ready.

Recommended current positioning:

```text
Experimental cash forecasting model
```

Not:

```text
Production-ready liquidity forecasting system
```

The model is useful for learning, portfolio demonstration, and early experimentation.

## 12. Limitations

Current limitations:

1. The dataset is synthetic.
2. Only 173 daily records are available.
3. Forecasting horizon is limited to next-day total amount.
4. No holiday calendar is included.
5. No month-end or quarter-end business calendar features are included.
6. No real cash inflow/outflow separation exists yet.
7. No currency-specific forecasting models are included.
8. No external economic or settlement-calendar data is included.
9. No confidence intervals are produced.
10. The improvement over baseline is modest.

## 13. Recommended Next Improvements

Recommended improvements:

1. Add more historical daily data.
2. Add weekday, month-end, and quarter-end indicators.
3. Add holiday and banking-calendar features.
4. Separate inflows and outflows.
5. Create currency-specific daily aggregates.
6. Add payment-type-specific daily aggregates.
7. Add high-value payment concentration features.
8. Tune Random Forest hyperparameters.
9. Try Gradient Boosting later.
10. Add prediction intervals or uncertainty bands.
11. Save the forecasting model artifact.
12. Add inference script for future daily forecast scoring.

## 14. Conclusion

The first cash forecasting model successfully demonstrates a practical forecasting workflow:

```text
→ transaction-level payments
→ daily aggregation
→ lag and rolling features
→ baseline forecasts
→ time-based train/test split
→ regression model training
→ model comparison
→ business interpretation
```

The Random Forest Regressor currently performs best, improving MAE by 3.22% over the 7-day moving average baseline.

The improvement is modest, but the workflow is correct. Future improvements should focus on stronger time-series features, more historical data, and richer business-calendar context.

The key applied AI lesson is:

```text
A forecasting model should be judged against simple business baselines, not in isolation.
```
