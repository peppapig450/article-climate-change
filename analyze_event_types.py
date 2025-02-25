from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from aggregate_and_visualize import (
    convert_damage,
    process_giss_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def compute_impact(
    df: pd.DataFrame, injury_weight: float, death_weight: float
) -> pd.Series:
    """Compute the impact for each event in the DataFrame."""
    return (
        convert_damage(df["damage_property"])
        + convert_damage(df["damage_crops"])
        + (df["injuries_direct"] + df["injuries_indirect"]) * injury_weight
        + (df["deaths_direct"] + df["deaths_indirect"]) * death_weight
    )


def aggregate_chunk(
    chunk: pd.DataFrame, injury_weight: float, death_weight: float
) -> pd.DataFrame:
    """Aggregate a single chunk by year and event_type."""
    chunk["impact"] = compute_impact(chunk, injury_weight, death_weight)
    return (
        chunk.groupby(["year", "event_type"])
        .agg(total_impact=("impact", "sum"), event_frequency=("event_type", "size"))
        .reset_index()
    )


def aggregate_by_event_type(
    csv_file: str,
    injury_weight: float = 200000,
    death_weight: float = 5000000,
) -> pd.DataFrame:
    """Aggregate NOAA data by year and event_type, computing total impact and frequency."""
    usecols = [
        "year",
        "event_type",
        "damage_property",
        "damage_crops",
        "injuries_direct",
        "injuries_indirect",
        "deaths_direct",
        "deaths_indirect",
    ]
    chunksize = 100_000
    agg_chunks: list[pd.DataFrame] = []

    logging.info(f"Processing NOAA data by event type: {csv_file}")
    for chunk in pd.read_csv(
        csv_file,
        usecols=usecols,
        low_memory=False,
        chunksize=chunksize,
        compression="zstd",
    ):
        if not isinstance(chunk, pd.DataFrame):
            continue
        aggregated = aggregate_chunk(chunk, injury_weight, death_weight)
        agg_chunks.append(aggregated)

    df = pd.concat(agg_chunks).groupby(["year", "event_type"]).sum().reset_index()
    return df


def compute_composite_index(
    df: pd.DataFrame, impact_weight: float = 0.5, frequency_weight: float = 0.5
) -> pd.DataFrame:
    """Compute normalized impact, frequency, and composite index."""
    df = df.assign(
        norm_total_impact=lambda x: x.groupby("event_type")["total_impact"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
        ),
        norm_event_frequency=lambda x: x.groupby("event_type")[
            "event_frequency"
        ].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
        ),
        composite_index=lambda x: (
            impact_weight * x["norm_total_impact"]
            + frequency_weight * x["norm_event_frequency"]
        ),
    )
    return df


def check_stationarity(series: pd.Series, max_diff: int = 2) -> tuple[pd.Series, int]:
    """Check stationarity with  ADF test and apply differencing if needed"""
    for diff in range(max_diff + 1):
        if diff > 0:
            test_series = series.diff().dropna()
        else:
            test_series = series.dropna()
        if len(test_series) < 5:  # Too few points for reliable ADF
            return series, 0
        result = adfuller(test_series, autolag="AIC")
        p_value = result[1]
        if p_value < 0.05:  # Stationary if p < 0.05
            return test_series, diff
        series = test_series
    return series, max_diff  # Return last differenced if still non-stationary


def compute_correlations(group: pd.DataFrame, lag_years: int) -> dict:
    """Compute various correlations for a single event type group."""
    valid_data = group.dropna(subset=["composite_index", "temp_anomaly"])
    valid_lagged = group.dropna(subset=["composite_index", "temp_anomaly_lagged"])

    if len(valid_data) < 2 or len(valid_lagged) < 2:
        return {}

    spearman_corr, spearman_p = spearmanr(
        valid_data["composite_index"], valid_data["temp_anomaly"]
    )
    pearson_corr, pearson_p = pearsonr(
        valid_data["composite_index"], valid_data["temp_anomaly"]
    )
    kendall_corr, kendall_p = kendalltau(
        valid_data["composite_index"], valid_data["temp_anomaly"]
    )
    lagged_corr, lagged_p = spearmanr(
        valid_lagged["composite_index"], valid_lagged["temp_anomaly_lagged"]
    )

    return {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "kendall_corr": kendall_corr,
        "kendall_p": kendall_p,
        "lagged_spearman_corr": lagged_corr,
        "lagged_spearman_p": lagged_p,
        "years_count": len(valid_data),
        "lagged_years_count": len(valid_lagged),
    }


def compute_granger_causality(
    group: pd.DataFrame, max_lag: int
) -> tuple[list[float], list[float], int, int]:
    """Compute Granger Causality p-values (forward and reverse) for multiple lags."""
    temp_series, temp_diff = check_stationarity(group["temp_anomaly"])
    comp_series, comp_diff = check_stationarity(group["composite_index"])
    granger_data = pd.DataFrame(
        {"temp_anomaly": temp_series, "composite_index": comp_series}
    ).dropna()

    granger_p_lags = [float("nan")] * max_lag
    rev_granger_p_lags = [float("nan")] * max_lag

    if len(granger_data) <= max_lag + 1:
        return granger_p_lags, rev_granger_p_lags, temp_diff, comp_diff

    try:
        # Forward: temp -> composite
        result = grangercausalitytests(
            granger_data[["temp_anomaly", "composite_index"]],
            maxlag=max_lag,
        )
        for lag in range(1, max_lag + 1):
            granger_p_lags[lag - 1] = result[lag][0]["ssr_chi2test"][1]

        # Reverse: composite -> temp
        result_rev = grangercausalitytests(
            granger_data[["composite_index", "temp_anomaly"]],
            maxlag=max_lag,
        )
        for lag in range(1, max_lag + 1):
            rev_granger_p_lags[lag - 1] = result_rev[lag][0]["ssr_chi2test"][1]
    except Exception as e:
        logging.debug(f"Granger test failed: {e}")

    return granger_p_lags, rev_granger_p_lags, temp_diff, comp_diff


def analyze_event_type_correlations(
    noaa_df: pd.DataFrame,
    giss_df: pd.DataFrame,
    min_years: int = 10,
    lag_years: int = 1,
    max_granger_lag: int = 2,
) -> pd.DataFrame:
    """
    Compute multiple correlation metrics and Granger Causality (forward and reverse) between each event type's
    composite index and temperature anomalies, with dynamic stationarity adjustment.
    """
    merged_df = pd.merge(noaa_df, giss_df, on="year", how="inner")
    if merged_df.empty:
        logging.warning("No overlapping data between NOAA and GISS datasets")
        return pd.DataFrame()

    merged_df = merged_df.assign(
        temp_anomaly_lagged=lambda x: x.groupby("event_type")["temp_anomaly"].shift(
            lag_years
        ),
    )

    correlations = []
    for event_type, group in merged_df.groupby("event_type"):
        if len(group) < min_years:
            continue

        corr_results = compute_correlations(group, lag_years)
        if not corr_results:
            continue

        granger_p_lags, rev_granger_p_lags, temp_diff, comp_diff = (
            compute_granger_causality(group, max_granger_lag)
        )
        granger_data = group.dropna(
            subset=["temp_anomaly", "composite_index"]
        )  # Use original for count

        result = {
            "event_type": event_type,
            **corr_results,
            "granger_years_count": len(granger_data),
            "temp_diff_order": temp_diff,
            "comp_diff_order": comp_diff,
        }
        for lag in range(1, max_granger_lag + 1):
            result[f"granger_p_lag{lag}"] = granger_p_lags[lag - 1]
            result[f"rev_granger_p_lag{lag}"] = rev_granger_p_lags[lag - 1]

        correlations.append(result)

    corr_df = pd.DataFrame(correlations).sort_values(
        by="spearman_corr", ascending=False
    )
    logging.info(
        "Top 10 event types by Spearman correlation with temperature anomalies:"
    )
    logging.info(
        corr_df[
            [
                "event_type",
                "spearman_corr",
                "spearman_p",
                "granger_p_lag1",
                "rev_granger_p_lag1years_count",
            ]
        ].head(10)
    )
    return corr_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlations and Granger Causality between NOAA event types and NASA GISS temperature anomalies."
    )
    parser.add_argument(
        "--csv-file",
        default="noaa_data/combined/combined_storm_events_1950-2024.csv.zst",
        help="Path to the combined NOAA CSV file",
    )
    parser.add_argument(
        "--giss-file",
        required=True,
        help="Path to the NASA GISS CSV file",
    )
    parser.add_argument(
        "--injury-weight",
        type=float,
        default=200000,
        help="Weight for injuries in impact calculation",
    )
    parser.add_argument(
        "--death-weight",
        type=float,
        default=5000000,
        help="Weight for deaths in impact calculation",
    )
    parser.add_argument(
        "--impact-weight",
        type=float,
        default=0.5,
        help="Weight for normalized total impact in composite index",
    )
    parser.add_argument(
        "--frequency-weight",
        type=float,
        default=0.5,
        help="Weight for normalized event frequency in composite index",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=20,
        help="Minimum number of years required for analysis",
    )
    parser.add_argument(
        "--lag-years",
        type=int,
        default=1,
        help="Number of years for lagged correlation",
    )
    parser.add_argument(
        "--max-granger-lag",
        type=int,
        default=2,
        help="Maximum lag for Granger Causality test",
    )
    args = parser.parse_args()

    csv_file = Path(args.csv_file).resolve()
    giss_file = Path(args.giss_file).resolve()
    if not csv_file.exists() or not giss_file.exists():
        logging.error(f"File not found: {csv_file} or {giss_file}")
        return

    giss_df = process_giss_data(str(giss_file))
    aggregated_df = aggregate_by_event_type(
        str(csv_file),
        args.injury_weight,
        args.death_weight,
    )
    noaa_df = compute_composite_index(
        aggregated_df, args.impact_weight, args.frequency_weight
    )

    corr_df = analyze_event_type_correlations(
        noaa_df,
        giss_df,
        min_years=args.min_years,
        lag_years=args.lag_years,
        max_granger_lag=args.max_granger_lag,
    )

    if not corr_df.empty:
        print(
            "Event types correlations and Granger Causality with temperature anomalies:"
        )
        pd.set_option("display.float_format", "{:.3f}".format)
        print(corr_df.to_string(index=False))
    else:
        print("No significant correlations or causality found.")


if __name__ == "__main__":
    main()
