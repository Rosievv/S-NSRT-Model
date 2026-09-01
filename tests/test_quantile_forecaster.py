import pandas as pd

from src.demand_forecasting import QuantileForecaster


def test_future_target_uses_exact_calendar_month_within_series() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"]),
            "hs_code": ["854231"] * 3,
            "country": ["Japan"] * 3,
            "value_usd": [10.0, 20.0, 40.0],
        }
    )

    supervised = QuantileForecaster(forecast_horizon=1).build_supervised_frame(frame)

    assert supervised["date"].tolist() == [pd.Timestamp("2020-01-01")]
    assert supervised["forecast_target"].tolist() == [20.0]


def test_automatic_features_exclude_current_and_future_targets() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "hs_code": ["854231"] * 6,
            "country": ["Japan"] * 6,
            "value_usd": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "known_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    model = QuantileForecaster(quantiles=[0.5], backend="sklearn", xgb_params={"n_estimators": 2})

    model.fit(frame)

    assert model.feature_cols == ["known_feature"]