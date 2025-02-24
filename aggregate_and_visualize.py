from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def convert_damage(damage_series: pd.Series) -> pd.Series:
    """Convert damage strings (e.g., '10.00K', '2.50M') to numeric values in dollars."""
    # Extract numeric value and unit
    extracted = (
        damage_series.fillna("0")
        .str.strip()
        .str.extract(r"([\d\.]+)([KM]?)")
        .fillna({"1": ""})
    )  # Ensure empty units are handled

    values = pd.to_numeric(extracted[0], errors="coerce").fillna(0)
    # Apply multiplies based on unit
    multipliers = extracted[1].map({"K": 1e3, "M": 1e6, "": 1}).fillna(1)

    return values * multipliers


def compute_event_impact(
    df: pd.DataFrame, injury_weight: float = 200000, death_weight: float = 5000000
) -> pd.Series:
    """
    Compute composite impact for each event in the DataFrame.
    The calculation sums property damage, crop damage, and weighted injuries and deaths.

    **Weight Justification:**
    - These weights use a 'Simplified Middle Ground' approach to balance economic
      realism with visual representation of human impacts (injuries, deaths) in visualizations.
    - Property and crop damages (in dollars) directly contribute to the economic impact.
    - Death weight ($5,000,000) is significantly higher than injury weight, reflecting the greater severity of loss of life.
    - Injury weight ($200,000) is set above average direct economic cost to ensure injuries are visually
      noticeable on plots, as economic damages can be orders of magnitude larger.
    - The goal is a visually informative representation where both human and economic dimensions
      are discernible in relation to temperature anomalies.
    - These weights are somewhat subjective and chosen for visual effectiveness, not strictly
      precise economic valuations. Further refinement may be needed based on visualization goals.
    """
    # Weight factors for computing composite event impact (justification explained in docstring):
    injury_weight = injury_weight  # $200,000 per injury (default, can be parameterized)
    death_weight = death_weight  # $5,000,000 per death (default, can be parameterized)

    # Convert property and crop damage strings to numeric values
    prop_damage = convert_damage(df["damage_property"])
    crop_damage = convert_damage(df["damage_crops"])

    impact_components = df.assign(
        # Calculate total injuries by summing direct and indirect injuries
        injuries=df["injuries_direct"].fillna(0) + df["injuries_indirect"].fillna(0),
        # Calculate total deaths by summing direct and indirect deaths
        deaths=df["deaths_direct"].fillna(0) + df["deaths_indirect"].fillna(0),
    ).pipe(
        # Use pipe to apply a function that calculates the final impact
        lambda df_assigned: prop_damage
        + crop_damage
        + (injury_weight * df_assigned["injuries"])
        + (death_weight * df_assigned["deaths"])
    )
    return impact_components


def aggregate_event_composite(
    csv_file: str,
    event_types_of_interest: list[str] | None = None,
    injury_weight: float = 200000,
    death_weight: float = 5000000,
    impact_weight: float = 0.5,
    frequency_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Process the NOAA CSV file and aggregate both total impact and event frequency by year.
    Then, create a composite index by normalizing each measure (using min-max scaling)
    and combining them as a weighted sum.

    Applies a 5-year moving average to smooth the event frequency data.

    Returns DataFrame with columns:
        - year
        - total_impact
        - event_frequency
        - norm_total_impact (Min-Max normalized total impact)
        - norm_event_frequency (Min-Max normalized event frequency)
        - composite_index (Weighted sum of normalized impact and frequency)
        - event_frequency_smoothed (5-year moving average of event frequency)

    Weights for composite index and impact calculation can be parameterized.
    Normalization method used is Min-Max scaling, which scales values to a [0, 1] range.
    A 5-year rolling average is applied to event frequency for smoothing temporal variations.

    **Weight Justification:**
    - The `injury_weight` and `death_weight` parameters used in the impact calculation are
      justified in the docstring of the `compute_event_impact` function.
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
    impact_aggregated = pd.Series(dtype=float)
    frequency_aggregated = pd.Series(dtype=float)

    logging.info(f"Processing NOAA data: {csv_file}")
    for chunk in pd.read_csv(
        csv_file,
        usecols=usecols,
        low_memory=False,
        chunksize=chunksize,
        compression="zstd",
    ):
        if not isinstance(chunk, pd.DataFrame):
            continue  # Handle cases where chunk might not be a DataFrame

        if event_types_of_interest:
            chunk = chunk[chunk["event_type"].isin(event_types_of_interest)]
            if chunk.empty:
                continue

        # Compute impact using parameterized weights
        chunk["impact"] = compute_event_impact(chunk, injury_weight, death_weight)

        # Aggregate impact and frequency by year, handling potential NaN years
        yearly_impact = chunk.groupby("year")["impact"].sum()
        yearly_freq = chunk.groupby("year").size()

        impact_aggregated = impact_aggregated.add(yearly_impact, fill_value=0)
        frequency_aggregated = frequency_aggregated.add(yearly_freq, fill_value=0)

    composite_df = (
        pd.DataFrame(
            {
                "year": impact_aggregated.index.astype(int),  # Ensure year is integer
                "total_impact": impact_aggregated.values,
                "event_frequency": frequency_aggregated.values,
            }
        )
        .sort_values("year")
        .set_index("year")
    )  # Sort by year and set as index

    # Normalization and Composite Index Calculation, and smoothing
    composite_df = composite_df.assign(
        norm_total_impact=lambda x: (x["total_impact"] - x["total_impact"].min())
        / (
            x["total_impact"].max() - x["total_impact"].min()
        ),  # Min-Max normalization for total impact
        norm_event_frequency=lambda x: (
            x["event_frequency"] - x["event_frequency"].min()
        )
        / (
            x["event_frequency"].max() - x["event_frequency"].min()
        ),  # Min-max normalization for total impact
        composite_index=lambda x: (
            impact_weight * x["norm_total_impact"]
            + frequency_weight * x["norm_event_frequency"]
        ),  # Weighted composite index
        event_frequency_smoothed=lambda x: x["event_frequency"]
        .rolling(window=5, min_periods=1)
        .mean(),  # 5-year moving average for smoothing
    ).reset_index()  # Reset index to make 'year' a column again

    return composite_df


def process_giss_data(giss_file: str) -> pd.DataFrame:
    """
    Process the NASA GISS Land-Ocean Temperature data file.
    Expects a CSV with columns 'Year' and 'J-D' (temperature anomaly).
    """
    try:
        giss_df = (
            pd.read_csv(
                giss_file,
                skiprows=1,
                usecols=["Year", "J-D"],  # Directly select relevant columns
                na_values="***",  # Handle potential '***' as NaN in source if it's used as missing as missing data
            )
            .rename(
                columns={"Year": "year", "J-D": "temp_anomaly"}
            )  # Rename columns after reading
            .dropna(
                subset=["year", "temp_anomaly"]
            )  # Drop rows with NaN in 'year' or 'temp_anomaly'
        )

        # Convert columns to numeric, errors='coerce' will turn non-numeric to NaN
        giss_df = giss_df.assign(
            year=pd.to_numeric(giss_df["year"], errors="coerce").astype(
                "int64"
            ),  # use int64 for nullable integers
            temp_anomaly=pd.to_numeric(giss_df["temp_anomaly"], errors="coerce"),
        ).dropna(
            subset=["year", "temp_anomaly"]
        )  # Drop any newly introduced NaNs from conversion

        return giss_df

    except Exception as e:
        logging.error(f"Error processing GISS file {giss_file}: {e}", exc_info=True)
        return pd.DataFrame(columns=["year", "temp_anomaly"])


def plot_dual_axis_composite(composite_df: pd.DataFrame, giss_df: pd.DataFrame):
    """
    Merge composite index data with temperature anomalies and plot a dual-axis chart.
    - Left axis: Temperature Anomaly, with dynamic thresholds to highlight notable anomalies.
    - Right axis: Composite index (combining impact and frequency).

    Computes Spearman rank correlation coefficient and performs a linear regression
    between temperature anomalies and the composite index. An inset scatter plot displays the regression line.
    """
    merged_df = pd.merge(composite_df, giss_df, on="year", how="inner")
    if merged_df.empty:
        logging.warning("No overlapping years between NOAA and GISS data.")
        return

    # Compute dynamic threshold: flag years where anomaly deviates more than one standard deviation from the mean
    mean_anomaly = merged_df["temp_anomaly"].mean()
    std_anomaly = merged_df["temp_anomaly"].std()
    notable_years = merged_df[
        abs(merged_df["temp_anomaly"] - mean_anomaly) > std_anomaly
    ]

    # Compute Spearman's rank correlation coefficient between temperature anomaly and composite index
    correlation_spearman = merged_df["temp_anomaly"].corr(
        merged_df["composite_index"], method="spearman"
    )

    # Perform linear regression analysis
    regression_result = linregress(
        merged_df["temp_anomaly"], merged_df["composite_index"]
    )
    reg_line = (
        regression_result.intercept
        + regression_result.slope * merged_df["temp_anomaly"]
    )  # type: ignore

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot temperature anomalies on the left axis
    ax1.plot(
        merged_df["year"],
        merged_df["temp_anomaly"],
        color="tab:red",
        marker="o",
        label="Temp Anomaly (°C)",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temperature Anomaly (°C)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Dynamically annotate notable temperature anomalies
    for _, row in notable_years.iterrows():
        ax1.annotate(
            f"{row['temp_anomaly']:.2f}",
            (row["year"], row["temp_anomaly"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="tab:red",
        )

    # Plot the composite index on the right axis
    ax2 = ax1.twinx()
    ax2.plot(
        merged_df["year"],
        merged_df["composite_index"],
        color="tab:blue",
        marker="s",
        label="Composite Index",
    )
    ax2.set_ylabel("Composite Index (Normalized)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Annotate the Spearman rank correlation coefficient on the chart
    ax1.text(
        0.05,
        0.95,
        f"Spearman ρ = {correlation_spearman:.2f}",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="black",
    )

    # Create an inset axes for the scatter plot with regression line (Linear regression kept for visual context)
    lin_reg_axes = (0.65, 0.15, 0.25, 0.25)
    ax_inset = fig.add_axes(lin_reg_axes)
    ax_inset.scatter(
        merged_df["temp_anomaly"],
        merged_df["composite_index"],
        color="purple",
        s=20,
        label="Data Points",
    )
    ax_inset.plot(
        merged_df["temp_anomaly"],
        reg_line,
        color="black",
        linestyle="--",
        label="Regression Line",
    )
    ax_inset.set_xlabel("Temp Anomaly")
    ax_inset.set_ylabel("Composite Index")
    ax_inset.legend(fontsize=8)
    ax_inset.grid(True, linestyle="--", alpha=0.5)

    plt.title(
        f"Global Temperature Anomalies vs. Extreme Weather Composite Index\n"
        f"(Linear Regression slope = {regression_result.slope:.2f})"  # Linear regression slope still relevant for visual trend # type: ignore
    )
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NOAA extreme weather impacts and frequency combined into a composite index with NASA GISS temperature anomalies."
    )
    parser.add_argument(
        "--csv-file",
        default="noaa_data/combined/combined_storm_events_1950-2024.csv.zst",
        help="Path to the combined NOAA CSV file",
    )
    parser.add_argument(
        "--giss-file", required=True, help="Path to the NASA GISS CSV file"
    )
    parser.add_argument(
        "--event-types",
        nargs="*",
        help="List of event types to filter (e.g., Flood Wildfire)",
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
    args = parser.parse_args()

    csv_file = Path(args.csv_file).resolve()
    if not csv_file.exists():
        logging.error(f"Combined CSV file not found: {csv_file}")
        return

    giss_file = Path(args.giss_file).resolve()
    if not giss_file.exists():
        logging.error(f"GISS file not found: {giss_file}")
        return

    composite_df = aggregate_event_composite(
        str(csv_file),
        args.event_types,
        args.injury_weight,
        args.death_weight,
        args.impact_weight,
        args.frequency_weight,
    )
    giss_df = process_giss_data(str(giss_file))
    plot_dual_axis_composite(composite_df, giss_df)


if __name__ == "__main__":
    main()
