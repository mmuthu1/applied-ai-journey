# Cash Forecast Review Runbook

## Purpose

This runbook explains how to interpret next-day cash forecast outputs.

The cash forecasting model predicts the next day’s total payment amount using daily payment activity, rolling averages, payment counts, failure rates, high-value payment rates, and calendar features.

## Model Used

Current forecasting model:

```text
cash_forecast_model
```

Current model type:

```text
Random Forest Regressor
```

## Forecast Outputs

The inference workflow returns:

    predicted next-day total amount
    forecast band
    difference from rolling 7-day average
    percentage difference from rolling 7-day average
    recommended action
    
## Forecast Bands

### LOW Forecast Band

Typical meaning:

```text
Lower expected payment activity
```

Operations guidance:

    Confirm whether lower activity is expected.
    Check for holidays, weekends, or business-calendar effects.
    Monitor for unexpected payment delays or reduced file volume.
    Confirm whether any inbound files are missing.

### MEDIUM Forecast Band

Typical meaning:

```text
Normal expected payment activity
```

Operations guidance:

    Continue normal cash operations planning.
    Monitor exception volume.
    No immediate escalation is required unless other risk signals are present.

### HIGH Forecast Band

Typical meaning:

```text
Higher expected payment activity
```

Operations guidance:

    Review liquidity needs.
    Review staffing and operational coverage.
    Monitor high-value payment queue.
    Prepare for higher exception or investigation volume.
    Check whether volume increase is expected due to month-end, quarter-end, or large client activity.

## Important Fields to Review
Operations should pay attention to:

    daily payment count
    daily total amount
    failed payment count
    high-value payment count
    failed payment rate
    high-value payment rate
    previous day total amount
    rolling 3-day average amount
    rolling 7-day average amount
    rolling payment counts
    day of week
    weekend indicator

## Business Interpretation

A forecast is an estimate, not a guarantee.

The forecast should support operational planning, liquidity awareness, and staffing decisions.

The current model should not be used for production liquidity decisions without additional validation.

## Current Limitations

This model is not production-ready.

Current limitations:

1. The model is trained on synthetic data.
2. The dataset contains only limited daily history.
3. The model does not include holidays.
4. The model does not include month-end or quarter-end indicators.
5. The model does not provide prediction intervals.
6. The model does not separately forecast inflows and outflows.
7. The model is not validated on real cash activity.

## Production Recommendations

Before production use, the following improvements are required:

1. Use longer historical time series.
2. Add banking holiday calendar features.
3. Add month-end and quarter-end indicators.
4. Separate inflows and outflows.
5. Add currency-specific forecasts.
6. Add prediction intervals.
7. Monitor forecast error over time.
8. Compare forecasts against actual daily payment totals.

