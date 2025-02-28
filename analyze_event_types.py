from __future__ import annotations

import argparse
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, permutation_test, spearmanr
from sklearn.utils import resample
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

from aggregate_and_visualize import (
    convert_damage,
    process_giss_data,
)
from visualize_event_types import launch_dashboard

if TYPE_CHECKING:
    from numpy import float64, int_
    from numpy.typing import ArrayLike, NDArray

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

# Minumum sample sizes for reliable analysis
MIN_SAMPLE_CORR: Final[int] = 10  # For correlation calculations, based on statistical rule of thumb
MIN_SAMPLE_GRANGER: Final[int] = 20  # # For Granger causality, stricter due to model complexity


def spearman_statistic(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute Spearman's correlation coefficient for permutation testing.

    Parameters
    ----------
    x : array_like
        First set of observations.
    y : array_like
        Second set of observations.

    Returns
    -------
    float
        Spearman's rank correlation coefficient.

    """
    return float(spearmanr(x, y)[0])


def bootstrap_ci(
    x: ArrayLike,
    y: ArrayLike,
    n_bootstraps: int = 1000,
    ci: float = 95,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[float, float]:
    """
    Compute confidence interval for Spearman's correlation using bootstrapping.

    This function estimates the variability of Spearman's rank correlation coefficient
    by resampling paired observations with replacement, suitable for small sample sizes.

    Parameters
    ----------
    x : array_like
        First array of observations (e.g., composite_index).
    y : array_like
        Second array of observations (e.g., temp_anomaly), paired with x.
    n_bootstraps : int, optional
        Number of bootstrap iterations (default: 1000).
    ci : float, optional
        Confidence level as a percentage (default: 95).
    random_state : int or np.random.RandomState, optional
        Seed or RandomState for reproducibility (default: None).

    Returns
    -------
    tuple
        Tuple of (lower, upper) confidence interval bounds for Spearman's rho.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., mismatched lengths, insufficient size,
        non-numeric data, or invalid parameters).

    """
    # Convert inputs to numpy arrays for consistency
    x = np.asarray(x)
    y = np.asarray(y)

    # Input validation
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError("Input arrays must have at least 2 elements for correlation")
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("x and y must contain numeric data")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("x and y must not contain NaN values")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")
    if not 0 < ci < 100:
        raise ValueError("ci must be between 0 and 100")

    # Bootstrap resampling
    boot_corrs = np.empty(n_bootstraps, dtype=float)
    rng = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
    for i in range(n_bootstraps):
        indices = resample(np.arange(n), replace=True, n_samples=n, random_state=rng)
        x_boot = x[indices]
        y_boot = y[indices]
        corr, _ = spearmanr(x_boot, y_boot)
        boot_corrs[i] = corr

    # Check if we have enough valid samples
    if len(boot_corrs) < 2:
        raise ValueError("Too few valid bootstrap samples to compute CI")

    # Compute percentiles
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)

    return lower, upper


def compute_aicc(aic_values: ArrayLike, n: int, lags: ArrayLike, m: int = 2) -> NDArray[float64]:
    """
    Compute AICc for VAR models, including intercepts.

    Parameters
    ----------
    aic_values : array_like
        Array of AIC values for lags 0 to max_lag.
    n : int
        Number of observations.
    lags : array_like
        Array of lag values corresponding to aic_values.
    m : int, optional
        Number of variables in the VAR model (default=2).

    Returns
    -------
    ndarray
        Array of AICc values.

    Notes
    -----
    AICc is calculated as AIC + 2 * k * (k + 1) / (n - k - 1), where k is the number
    of parameters. For VAR models, k = lags * m**2 + m (including intercepts).

    """
    aic_values_array: NDArray[float64] = np.asarray(aic_values, dtype=np.float64)
    lags_array: NDArray[int_] = np.asarray(lags, dtype=np.int_)
    k: NDArray[int_] = lags_array * m**2 + m  # AR coefficients + intercepts

    # Prevent division by zero or negative denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        penalty: NDArray[float64] = np.where(n > k + 1, 2 * k * (k + 1) / (n - k - 1), np.float64(np.inf))
    return aic_values_array + penalty


def compute_impact(df: pd.DataFrame, injury_weight: float, death_weight: float) -> pd.Series:
    """
    Compute the economic impact for each event in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for damage, injuries, and deaths.
    injury_weight : float
        Economic weight per injury (default: 50000). Based on rough estimates of
        medical costs and lost productivity; adjust per studies (e.g., DOT data).
    death_weight : float
        Economic weight per death (default: 7600000). Based on FEMA's 2023 statistical
        value of a human life for cost-benefit analysis in disaster mitigation.

    Returns
    -------
    pd.Series
        Series of impact values combining property/crop damages, injuries, and deaths.

    Notes
    -----
    The impact is calculated as:
    impact = damage_property + damage_crops + (injuries_direct + injuries_indirect) * injury_weight
             + (deaths_direct + deaths_indirect) * death_weight
    where damages are converted using `convert_damage`.

    """
    return (
        convert_damage(df["damage_property"])
        + convert_damage(df["damage_crops"])
        + (df["injuries_direct"] + df["injuries_indirect"]) * injury_weight
        + (df["deaths_direct"] + df["deaths_indirect"]) * death_weight
    )


def aggregate_by_event_type(
    csv_file: str,
    injury_weight: float = 50000,
    death_weight: float = 7600000,
) -> pd.DataFrame:
    """
    Aggregate NOAA data by year and event_type efficiently using a running total.

    This function processes large datasets in chunks, accumulating totals in a defaultdict
    to avoid multiple concatenations and groupby operations, improving memory and time efficiency.

    Parameters
    ----------
    csv_file : str
        Path to the NOAA CSV file.
    injury_weight : float, optional
        Economic weight per injury (default=50000).
    death_weight : float, optional
        Economic weight per death (default=7600000).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns: year, event_type, total_impact, event_frequency.

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

    df = pd.DataFrame.from_dict(agg_dict, orient="index").reset_index(
        names=["year", "event_type", "total_impact", "event_frequency"]
    )

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

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'total_impact' and 'event_frequency' columns.
    impact_weight : float, optional
        Weight for normalized total impact (default=0.5).
    frequency_weight : float, optional
        Weight for normalized event frequency (default=0.5).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - norm_total_impact: Normalized total impact per event type.
        - norm_event_frequency: Normalized event frequency per event type.
        - composite_index: Weighted sum of norm_total_impact and norm_event_frequency.

    Notes
    -----
    Normalization is done per event type using min-max scaling.
    If the range is zero, the normalized value is set to 0.

    """
    df = df.assign(
        norm_total_impact=lambda x: x.groupby("event_type")["total_impact"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
        ),
        norm_event_frequency=lambda x: x.groupby("event_type")["event_frequency"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
        ),
        composite_index=lambda x: (
            impact_weight * x["norm_total_impact"] + frequency_weight * x["norm_event_frequency"]
        ),
    )
    return df


def check_stationarity(series: pd.Series, max_diff: int = 2, min_len: int = 5) -> tuple[pd.Series, int]:
    """
    Check stationarity using the Augmented Dickey-Fuller test and apply differencing if necessary.

    Parameters
    ----------
    series : pd.Series
        Time series to test for stationarity.
    max_diff : int, optional
        Maximum number of differencing attempts (default=2).
    min_len : int, optional
        Minimum length of the series for ADF test (default=5).

    Returns
    -------
    tuple
        - differenced_series : pd.Series
            The differenced series if non-stationary, else the original series.
        - diff_order : int
            Number of differences applied to achieve stationarity.

    Notes
    -----
    The function applies differencing up to `max_diff` times until the series is stationary
    (p-value < 0.05 in ADF test) or the maximum differencing is reached.
    If the series length is less than `min_len` after differencing, it returns the original series.

    """
    original_index = series.index
    for diff in range(max_diff + 1):
        test_series = series.diff().dropna() if diff > 0 else series.dropna()
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


def compute_correlations(group: pd.DataFrame, lag_years: int, min_size: int = 10) -> dict:
    """
    Compute various correlation metrics between composite index and temperature anomalies.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing 'composite_index', 'temp_anomaly', and 'temp_anomaly_lagged'.
    lag_years : int
        Number of years for lagged correlation.
    min_size : int, optional
        Minimum sample size for reliable correlation (default=MIN_SAMPLE_CORR).

    Returns
    -------
    dict
        Dictionary containing:
        - 'spearman_corr': Spearman's correlation coefficient
        - 'spearman_p': p-value for Spearman's correlation
        - 'spearman_p_perm': p-value from permutation test
        - 'spearman_ci_lower': Lower bound of confidence interval
        - 'spearman_ci_upper': Upper bound of confidence interval
        - 'pearson_corr': Pearson's correlation coefficient
        - 'pearson_p': p-value for Pearson's correlation
        - 'kendall_corr': Kendall's tau correlation coefficient
        - 'kendall_p': p-value for Kendall's tau
        - 'lagged_spearman_corr': Spearman's correlation with lagged temperature
        - 'lagged_spearman_p': p-value for lagged Spearman's correlation
        - 'years_count': Number of years with valid data
        - 'lagged_years_count': Number of years with valid lagged data
        - 'impact_spearman_corr': Spearman's correlation for normalized impact
        - 'impact_spearman_p': p-value for impact correlation
        - 'frequency_spearman_corr': Spearman's correlation for normalized frequency
        - 'frequency_spearman_p': p-value for frequency correlation

    Notes
    -----
    - Requires at least 2 valid data points.
    - If sample size < min_size, a warning is logged.
    - Uses permutation test for Spearman's p-value.

    """
    valid_data = group.dropna(subset=["composite_index", "temp_anomaly"])
    valid_lagged = group.dropna(subset=["composite_index", "temp_anomaly_lagged"])

    if len(valid_data) < 2 or len(valid_lagged) < 2:
        return {}

    # Warning for small sample sizes
    if len(valid_data) < min_size:
        logging.warning(
            f"{group['event_type'].iloc[0]}: Sample size {len(valid_data)} < {min_size}. Results may be unreliable."
        )

    perm_result = permutation_test(
        (valid_data["composite_index"], valid_data["temp_anomaly"]),
        statistic=spearman_statistic,
        permutation_type="pairings",
        n_resamples=999,  # Reasonable default for approximation; adjust as needed
        alternative="two-sided",
    )
    spearman_corr = perm_result.statistic
    spearman_p_perm = perm_result.pvalue

    # Bootstrap CI
    spearman_lower, spearman_upper = bootstrap_ci(valid_data["composite_index"], valid_data["temp_anomaly"])

    spearman_corr, spearman_p = spearmanr(valid_data["composite_index"], valid_data["temp_anomaly"])
    pearson_corr, pearson_p = pearsonr(valid_data["composite_index"], valid_data["temp_anomaly"])
    kendall_corr, kendall_p = kendalltau(valid_data["composite_index"], valid_data["temp_anomaly"])
    lagged_corr, lagged_p = spearmanr(valid_lagged["composite_index"], valid_lagged["temp_anomaly_lagged"])

    impact_spearman_corr, impact_spearman_p = spearmanr(valid_data["norm_total_impact"], valid_data["temp_anomaly"])
    frequency_spearman_corr, frequency_spearman_p = spearmanr(
        valid_data["norm_event_frequency"], valid_data["temp_anomaly"]
    )

    return {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "spearman_p_perm": spearman_p_perm,
        "spearman_ci_lower": spearman_lower,
        "spearman_ci_upper": spearman_upper,
        "pearson_corr": pearson_corr,
        "pearson_p": pearson_p,
        "kendall_corr": kendall_corr,
        "kendall_p": kendall_p,
        "lagged_spearman_corr": lagged_corr,
        "lagged_spearman_p": lagged_p,
        "years_count": len(valid_data),
        "lagged_years_count": len(valid_lagged),
        "impact_spearman_corr": impact_spearman_corr,
        "impact_spearman_p": impact_spearman_p,
        "frequency_spearman_corr": frequency_spearman_corr,
        "frequency_spearman_p": frequency_spearman_p,
    }


def compute_granger_causality(
    group: pd.DataFrame, max_lag: int
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], int, int]:
    """
    Compute Granger Causality p-values in both directions for optimal lags selected by AICc and BIC.

    Steps:
        1. Check stationarity and apply differencing if necessary.
        2. Fit a VAR model for lags 1 to max_lag and compute AIC and BIC.
        3. Compute AICc from AIC values.
        4. Select optimal lags based on AICc and BIC.
        5. Run Granger causality tests for optimal lags > 0.
        6. Return p-values for forward (temp -> composite) and reverse causality.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing 'year', 'temp_anomaly', and 'composite_index'.
    max_lag : int
        Maximum lag to test in VAR and Granger causality.

    Returns
    -------
    tuple
        - forward_p_values_by_criterion : dict
            P-values for forward Granger causality (temp -> composite) by criterion.
        - reverse_p_values_by_criterion : dict
            P-values for reverse Granger causality (composite -> temp) by criterion.
        - temp_diff : int
            Differencing order applied to temperature series.
        - comp_diff : int
            Differencing order applied to composite index series.

    Notes
    -----
    - Requires consecutive years without gaps.
    - Handles non-stationarity by differencing up to 2 times.
    - Uses AICc and BIC for lag selection.

    """
    event_type = group["event_type"].iloc[0]
    years = group["year"].sort_values().unique()
    year_diffs = np.diff(years)
    if np.any(year_diffs > 1):
        logging.warning(
            f"{event_type}: Skipping Granger causality due to non-consecutive years (gaps: {year_diffs[year_diffs > 1]})"
        )
        test_types = ["params_ftest", "ssr_ftest"]
        return (
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            0,
            0,
        )

    # Early exit if data is all NaN
    if group["temp_anomaly"].isna().all() or group["composite_index"].isna().all():
        logging.debug(f"{group['event_type'].iloc[0]}: All NaN in input series")
        test_types = ["params_ftest", "ssr_ftest"]
        return (
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            0,
            0,
        )

    # Set up time-indexed series for stationarity checks
    group_indexed = group.set_index(pd.to_datetime(group["year"], format="%Y")).asfreq("YS")
    temp_series, temp_diff = check_stationarity(group_indexed["temp_anomaly"], max_diff=2, min_len=max_lag + 1)
    comp_series, comp_diff = check_stationarity(group_indexed["composite_index"], max_diff=2, min_len=max_lag + 1)

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
        test_types = ["params_ftest", "ssr_ftest"]
        return (
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            0,
            0,
        )

    try:
        # Fit VAR model to select optimal lags
        model = VAR(granger_data)
        lags = np.arange(0, max_lag + 1)
        aic_vals = [float("inf")] + [model.fit(int(lag)).aic for lag in range(1, max_lag + 1)]
        bic_vals = [float("inf")] + [model.fit(int(lag)).bic for lag in range(1, max_lag + 1)]
        aicc_vals = compute_aicc(np.array(aic_vals), n, lags, m=2)

        # Select optimal lags, excluding lag 0
        optimal_lags = {
            "aicc": int(np.argmin(aicc_vals[1:]) + 1) if np.isfinite(aicc_vals[1:]).any() else 0,
            "bic": int(np.argmin(bic_vals[1:]) + 1) if np.isfinite(bic_vals[1:]).any() else 0,
        }
        logging.info(f"{group['event_type'].iloc[0]}: Optimal lags = {optimal_lags}")

        # Initialize p-value dictionaries with nested structure for each test type
        test_types = ["params_ftest", "ssr_ftest"]
        forward_p_values_by_criterion = {
            crit: {test: [float("nan")] * max_lag for test in test_types} for crit in optimal_lags
        }
        reverse_p_values_by_criterion = {
            crit: {test: [float("nan")] * max_lag for test in test_types} for crit in optimal_lags
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
                            result[lag_order][0]["params_ftest"][1] if lag_order in result else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                        "ssr_ftest": [
                            result[lag_order][0]["ssr_ftest"][1] if lag_order in result else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                    }
                    reverse_p_values_by_criterion[crit] = {
                        "params_ftest": [
                            result[lag_order][0]["params_ftest"][1] if lag_order in result_rev else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                        "ssr_ftest": [
                            result[lag_order][0]["ssr_ftest"][1] if lag_order in result_rev else float("nan")
                            for lag_order in range(1, max_lag + 1)
                        ],
                    }
    except ValueError as e:
        logging.debug(f"Granger test failed due to VAR fitting error (e.g., singular matrix): {e!s}")
        test_types = ["params_ftest", "ssr_ftest"]
        return (
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            0,
            0,
        )

    except Exception as e:
        logging.debug(f"Granger test failed due to unexpected error: {e!s}")
        test_types = ["params_ftest", "ssr_ftest"]
        return (
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            {"aicc": {test: [float("nan")] * max_lag for test in test_types}},
            0,
            0,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute multiple correlation metrics and Granger Causality between each event type's
    composite index and temperature anomalies.

    Parameters
    ----------
    noaa_df : pd.DataFrame
        NOAA data with composite index.
    giss_df : pd.DataFrame
        NASA GISS temperature anomalies.
    min_years : int, optional
        Minimum number of years for analysis (default=10).
    lag_years : int, optional
        Lag years for correlation (default=1).
    max_granger_lag : int, optional
        Maximum lag for Granger Causality (default=2).

    Returns
    -------
    tuple
        - corr_df : pd.DataFrame
            Correlation and Granger Causality results.
        - merged_df : pd.DataFrame
            Merged NOAA and GISS data.

    """
    merged_df = noaa_df.merge(giss_df, on="year", how="inner")
    if merged_df.empty:
        logging.warning("No overlapping data between NOAA and GISS datasets")
        return pd.DataFrame(), pd.DataFrame()

    unlagged_merged_df = merged_df.copy(deep=True)

    merged_df = merged_df.assign(
        temp_anomaly_lagged=lambda x: x.groupby("event_type")["temp_anomaly"].shift(lag_years),
    )

    correlations = []
    for event_type, group in merged_df.groupby("event_type"):
        years = group["year"].sort_values().unique()
        if len(years) < 2 or len(group) < min_years:
            continue

        corr_results = compute_correlations(group, lag_years)
        if not corr_results:
            continue

        granger_results, rev_granger_results, temp_diff, comp_diff = compute_granger_causality(group, max_granger_lag)
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
                    strict=False,
                ):
                    result[f"{crit}_{test_type}_p_lag{lag}"] = forward_p_value
                    result[f"{crit}_rev_{test_type}_p_lag{lag}"] = reverse_p_value

        correlations.append(result)

    corr_df = pd.DataFrame(correlations).sort_values(by="spearman_corr", ascending=False)

    # Highlight significant Granger results
    significant = []
    for _, row in corr_df.iterrows():
        for col in corr_df.columns:
            if "p_lag" in col and pd.notna(row[col]) and row[col] < 0.05:
                direction = "temp -> event" if "rev_" not in col else "event -> temp"
                lag = int(col.split("_p_lag")[1])
                test = col.split("_")[1] if "rev_" not in col else col.split("_")[2]
                crit = col.split("_")[0]
                significant.append(
                    f"{row['event_type']}: {direction}, lag {lag}, {crit}_{test}_p = {row[col]:.3f}, n = {row['granger_years_count']}"
                )

    if significant:
        logging.info("Significant Granger casuality results (p < 0.05):")
        for sig in significant:
            logging.info(sig)

    logging.info("Top 10 event types by Spearman correlation with temperature anomalies:")
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
    return corr_df, unlagged_merged_df


def main() -> None:
    """
    Main function to run the analysis script.

    Parses command-line arguments, processes data, computes correlations and Granger Causality,
    and launches a dashboard for visualization.
    """
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
    noaa_df = compute_composite_index(aggregated_df, args.impact_weight, args.frequency_weight)

    corr_df, merged_df = analyze_event_type_correlations(
        noaa_df,
        giss_df,
        min_years=args.min_years,
        lag_years=args.lag_years,
        max_granger_lag=args.max_granger_lag,
    )

    if not corr_df.empty:
        logging.info("Full results saved to 'data/output/event_type_analysis.csv'")
        pd.set_option("display.float_format", "{:.3f}".format)
        corr_df.to_csv("data/output/event_type_analysis.csv", index=False)
        logging.info("Summary of correlations and Granger Causality (lag 1 only):")
        logging.info(
            corr_df[
                [
                    "event_type",
                    "spearman_corr",
                    "spearman_p",
                    "impact_spearman_corr",
                    "frequency_spearman_corr",
                    "granger_years_count",
                    "aicc_params_ftest_p_lag1",
                    "aicc_ssr_ftest_p_lag1",
                    "aicc_rev_params_ftest_p_lag1",
                    "aicc_rev_ssr_ftest_p_lag1",
                ]
            ].to_string(index=False)
        )
        launch_dashboard(merged_df, corr_df)
    else:
        logging.warning("No significant correlations or causality found.")


if __name__ == "__main__":
    main()
