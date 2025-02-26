from __future__ import annotations

import argparse
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from aggregate_and_visualize import (
    convert_damage,
    process_giss_data,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Suppress FutureWarnings from statsmodels cautiously
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

# Define logging level mapping for command-line flexibility
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def compute_aicc(aic_values: NDArray, n: int, lags: NDArray, m=2) -> NDArray:
    """
    Compute AICc for VAR models, including intercepts.

    Parameters:
        aic_values (NDArray): Array of AIC values for lags 0 to max_lag.
        n (int): Number of observations.
        lags (NDArray): Array of lag values corresponding to aic_values.
        m (int): Number of variables in the VAR model (default=2).

    Returns:
        NDArray: Array of AICc values.
    """
    k = lags * m**2 + m  # AR coefficients + intercepts
    penalty = np.where(n > k + 1, 2 * k * (k + 1) / (n - k - 1), np.inf)
    return aic_values + penalty


def compute_impact(
    df: pd.DataFrame, injury_weight: float, death_weight: float
) -> pd.Series:
    """
    Compute the impact for each event in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with damage, injury, and death columns.
        injury_weight (float): Economic weight per injury.
        death_weight (float): Economic weight per death.

    Returns:
        pd.Series: Impact values combining damages, injuries, and deaths.
    """
    return (
        convert_damage(df["damage_property"])
        + convert_damage(df["damage_crops"])
        + (df["injuries_direct"] + df["injuries_indirect"]) * injury_weight
        + (df["deaths_direct"] + df["deaths_indirect"]) * death_weight
    )


def aggregate_by_event_type(
    csv_file: str,
    injury_weight: float = 200000,
    death_weight: float = 5000000,
) -> pd.DataFrame:
    """
    Aggregate NOAA data by year and event_type efficiently using a running total.

    This function processes large datasets in chunks, accumulating totals in a defaultdict
    to avoid multiple concatenations and groupby operations, improving memory and time efficiency.

    Parameters:
        csv_file (str): Path to the NOAA CSV file.
        injury_weight (float): Economic weight per injury (default=200000).
        death_weight (float): Economic weight per death (default=5000000).

    Returns:
        pd.DataFrame: Aggregated DataFrame with year, event_type, total_impact, and event_frequency.
    """
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
    agg_dict = defaultdict(lambda: {"total_impact": 0.0, "event_frequency": 0})

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
        chunk["impact"] = compute_impact(chunk, injury_weight, death_weight)
        aggregated = (
            chunk.groupby(["year", "event_type"], observed=True)
            .agg(total_impact=("impact", "sum"), event_frequency=("event_type", "size"))
            .reset_index()
        )

        for row in aggregated.itertuples(index=False):
            key = (row.year, row.event_type)
            agg_dict[key]["total_impact"] += cast(float, row.total_impact)
            agg_dict[key]["event_frequency"] += cast(int, row.event_frequency)

    df = pd.DataFrame.from_dict(agg_dict, orient="index").reset_index()
    df.columns = ["year", "event_type", "total_impact", "event_frequency"]
    year_counts = df.groupby("event_type")["year"].nunique()
    logging.info("Event types and year counts before filtering:")
    logging.info(year_counts)
    return df


def compute_composite_index(
    df: pd.DataFrame, impact_weight: float = 0.5, frequency_weight: float = 0.5
) -> pd.DataFrame:
    """
    Compute normalized impact, frequency, and composite index for each event type.

    The composite index is a weighted sum of normalized total impact and event frequency.
    Default weights are 0.5 each, assuming equal importance. Adjust based on domain knowledge
    (e.g., increase impact_weight if economic damage is prioritized over frequency).

    Parameters:
        df (pd.DataFrame): DataFrame with total_impact and event_frequency.
        impact_weight (float): Weight for normalized impact (default=0.5).
        frequency_weight (float): Weight for normalized frequency (default=0.5).

    Returns:
        pd.DataFrame: DataFrame with added norm_total_impact, norm_event_frequency, and composite_index.
    """
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


def check_stationarity(
    series: pd.Series, max_diff: int = 2, min_len: int = 5
) -> tuple[pd.Series, int]:
    """
    Check stationarity with ADF test and apply differencing if needed up to max_diff times.

    Parameters:
        series (pd.Series): Time series to test.
        max_diff (int): Maximum number of differencing attempts (default=2).
        min_len (int): Minimum length for ADF test (default=5).

    Returns:
        tuple: (differenced series, number of differences applied)
    """
    original_index = series.index
    for diff in range(max_diff + 1):
        if diff > 0:
            test_series = series.diff().dropna()
        else:
            test_series = series.dropna()
        if len(test_series) < min_len:  # Too few points for reliable ADF
            return series, 0
        result = adfuller(test_series, autolag="AIC")
        p_value = result[1]
        logging.debug(f"Diff {diff}: p-value = {p_value:.3f}")
        if p_value < 0.05:  # Stationary if p < 0.05
            return test_series.reindex(original_index, fill_value=np.nan), diff
        series = test_series
    return series.reindex(
        original_index, fill_value=np.nan
    ), max_diff  # Return last differenced if still non-stationary


def compute_correlations(group: pd.DataFrame, lag_years: int) -> dict:
    """
    Compute various correlation metrics for a single event type group.

    Parameters:
        group (pd.DataFrame): DataFrame with composite_index, temp_anomaly, and temp_anomaly_lagged.
        lag_years (int): Number of years for lagged correlation.

    Returns:
        dict: Correlation coefficients and p-values.
    """
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
) -> tuple[
    dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], int, int
]:
    """
    Compute Granger Causality p-values in both directions for optimal lags selected by AICc and BIC.

    Steps:
        1. Check stationarity and apply differencing if necessary.
        2. Fit a VAR model for lags 1 to max_lag and compute AIC and BIC.
        3. Compute AICc from AIC values.
        4. Select optimal lags based on AICc and BIC.
        5. Run Granger causality tests for optimal lags > 0.
        6. Return p-values for forward (temp -> composite) and reverse causality.

    Note: Granger causality assumes stationarity and linearity. Non-stationary series after
    max differencing or non-linear relationships may invalidate results.

    Parameters:
        group (pd.DataFrame): DataFrame with 'year', 'temp_anomaly', and 'composite_index'.
        max_lag (int): Maximum lag to test in VAR and Granger causality.

    Returns:
        tuple: (forward p-values by criterion, reverse p-values by criterion, temp diff order, composite diff order)
    """
    # Early exit if data is all NaN
    if group["temp_anomaly"].isna().all() or group["composite_index"].isna().all():
        logging.debug(f"{group['event_type'].iloc[0]}: All NaN in input series")
        return (
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            0,
            0,
        )

    # Set up time-indexed series for stationarity checks
    group_indexed = group.set_index(pd.to_datetime(group["year"], format="%Y"))
    temp_series, temp_diff = check_stationarity(
        group_indexed["temp_anomaly"], max_diff=2, min_len=max_lag + 1
    )
    comp_series, comp_diff = check_stationarity(
        group_indexed["composite_index"], max_diff=2, min_len=max_lag + 1
    )

    # Prepare data for VAR and Granger tests
    granger_data = pd.DataFrame(
        {
            "temp_anomaly": temp_series,
            "composite_index": comp_series,
        },
    ).dropna()

    n = len(granger_data)
    if n <= max_lag + 1:
        logging.debug(f"Sample size too small: {n} <= {max_lag + 1}")
        return (
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            temp_diff,
            comp_diff,
        )

    try:
        # Fit VAR model to select optimal lags
        model = VAR(granger_data)
        lags = np.arange(0, max_lag + 1)
        aic_vals = [float("inf")] + [
            model.fit(int(lag)).aic for lag in range(1, max_lag + 1)
        ]
        bic_vals = [float("inf")] + [
            model.fit(int(lag)).bic for lag in range(1, max_lag + 1)
        ]
        aicc_vals = compute_aicc(np.array(aic_vals), n, lags, m=2)

        # Select optimal lags, excluding lag 0
        optimal_lags = {
            "aicc": int(np.argmin(aicc_vals[1:]) + 1)
            if np.isfinite(aicc_vals[1:]).any()
            else 0,
            "bic": int(np.argmin(bic_vals[1:]) + 1)
            if np.isfinite(bic_vals[1:]).any()
            else 0,
        }
        logging.info(f"{group['event_type'].iloc[0]}: Optimal lags = {optimal_lags}")

        # Initialize p-value dictionaries with nested structure for each test type
        test_types = ["params_ftest", "ssr_ftest"]
        forward_p_values_by_criterion = {
            crit: {test: [float("nan")] * max_lag for test in test_types}
            for crit in optimal_lags
        }
        reverse_p_values_by_criterion = {
            crit: {test: [float("nan")] * max_lag for test in test_types}
            for crit in optimal_lags
        }

        # Run Granger causality tests only for optimal lags > 0
        tested_lags = {lag for lag in optimal_lags.values() if lag > 0}
        for lag in tested_lags:
            # Forward: Does temp_anomaly Granger-cause composite_index?
            result = grangercausalitytests(
                granger_data[["temp_anomaly", "composite_index"]],
                maxlag=lag,
                verbose=False,
            )
            # Reverse: Does composite_index Granger-cause temp_anomaly?
            result_rev = grangercausalitytests(
                granger_data[["composite_index", "temp_anomaly"]],
                maxlag=lag,
                verbose=False,
            )
            for crit, opt_lag in optimal_lags.items():
                if opt_lag == lag:
                    forward_p_values_by_criterion[crit] = {
                        "params_ftest": [
                            result[lag_order][0]["params_ftest"][1]
                            if lag_order in result
                            else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                        "ssr_ftest": [
                            result[lag_order][0]["ssr_ftest"][1]
                            if lag_order in result
                            else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                    }
                    reverse_p_values_by_criterion[crit] = {
                        "params_ftest": [
                            result[lag_order][0]["params_ftest"][1]
                            if lag_order in result_rev
                            else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                        "ssr_ftest": [
                            result[lag_order][0]["ssr_ftest"][1]
                            if lag_order in result_rev
                            else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                    }
    except ValueError as e:
        logging.debug(
            f"Granger test failed due to VAR fitting error (e.g., singular matrix): {str(e)}"
        )
        return (
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            temp_diff,
            comp_diff,
        )
    except Exception as e:
        logging.debug(f"Granger test failed due to unexpected error: {str(e)}")
        return (
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            {
                "aicc": {
                    "params_ftest": [float("nan")] * max_lag,
                    "ssr_ftest": [float("nan")] * max_lag,
                }
            },
            temp_diff,
            comp_diff,
        )

    return (
        forward_p_values_by_criterion,
        reverse_p_values_by_criterion,
        temp_diff,
        comp_diff,
    )


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
        years = group["year"].sort_values().unique()
        if len(years) < 2:
            continue
        if np.any((years_diff := np.diff(years)) > 1):
            logging.warning(
                f"{event_type}: Non-consecutive years detected, which may affect time series analysis"
            )

        if len(group) < min_years:
            continue

        corr_results = compute_correlations(group, lag_years)
        if not corr_results:
            continue

        granger_results, rev_granger_results, temp_diff, comp_diff = (
            compute_granger_causality(group, max_granger_lag)
        )
        granger_data = group.dropna(subset=["temp_anomaly", "composite_index"])

        result = {
            "event_type": event_type,
            **corr_results,
            "granger_years_count": len(granger_data),
            "temp_diff_order": temp_diff,
            "comp_diff_order": comp_diff,
        }
        for crit in granger_results:
            for test_type in granger_results[crit]:
                for lag, forward_p_value, reverse_p_value in zip(
                    range(1, max_granger_lag + 1),
                    granger_results[crit][test_type],
                    rev_granger_results[crit][test_type],
                ):
                    result[f"{crit}_{test_type}_p_lag{lag}"] = forward_p_value
                    result[f"{crit}_rev_{test_type}_p_lag{lag}"] = reverse_p_value

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
                "aicc_params_ftest_p_lag1",
                "aicc_ssr_ftest_p_lag1",
                "aicc_rev_params_ftest_p_lag1",
                "aicc_rev_ssr_ftest_p_lag1",
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS.keys(),
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger().setLevel(LOG_LEVELS[args.log_level])

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
