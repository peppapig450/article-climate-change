from collections.abc import Callable

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


def launch_dashboard(merged_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """Launch an interactive Dash dashboard with multiple visualizations."""
    app = dash.Dash(__name__)

    # Type-safe Dash callback decorator
    def dash_callback[F: Callable[..., go.Figure]](output: Output, inputs: list[Input]) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            app.callback(output, inputs)(func)  # Registers with Dash inside the decorator
            return func

        return decorator

    # Extract unique event types for dropdown menus
    event_types = merged_df["event_type"].unique()

    # Define the dashboard layout with a clear title and introductory text
    app.layout = html.Div(
        [
            html.H1("Climate Change and Storm Events: An Interactive Exploration"),
            html.P(
                "Explore how temperature changes relate to storm impacts over time. "
                "Use the dropdowns to select specific storm types and see trends, correlations, and causality insights."
            ),
            dcc.Tabs(
                [
                    # Tab 1:  Time Series - Enhanced for public understanding
                    dcc.Tab(
                        label="Trends Over Time",
                        children=[
                            html.H3("Temperature and Storm Impact Trends"),
                            html.P(
                                "See how temperature changes and storm impacts have evolved over the years. "
                                "The 'Temperature Change' line shows deviations from average temperatures, "
                                "while the 'Storm Impact Index' combines damage and frequency for each storm type."
                            ),
                            dcc.Dropdown(
                                id="event-dropdown",
                                options=[{"label": event, "value": event} for event in event_types],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="time-series-plot"),
                        ],
                    ),
                    # Tab 2: Correlation Scatter - Simplified for public, detailed for experts
                    dcc.Tab(
                        label="Temperature vs. Storm Impact",
                        children=[
                            html.H3("How Temperature relates to Storm Impacts."),
                            html.P(
                                "This scatter plot shows the relationship between temperature changes and storm impacts. ",
                                "A trendline indicates the overall pattern. The annotation provides a measure of how strongly ",
                                "they are linked (correlation) and the likelihood this link is real (chance likelihood). ",
                                "Hover over points for details.",
                            ),
                            dcc.Dropdown(
                                id="scatter-event-dropdown",
                                options=[
                                    {"label": event, "value": event} for event in merged_df["event_type"].unique()
                                ],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="scatter-plot"),
                        ],
                    ),
                    # Tab 3: Granger Causality - Enhanced for expert analysis
                    dcc.Tab(
                        label="Causality Insights",
                        children=[
                            html.H3("Does Temperature Change Predict Storm Impacts?"),
                            html.P(
                                "This bar plot shows the likelihood that temperature changes help predict future storm impacts "
                                "(forward casuality) and vice versa (reverse casuality). Bars below the red line (p < 0.05) suggest ",
                                "a significant predictive relationship. Select different lags to see how delayed effects might play a role.",
                            ),
                            dcc.Dropdown(
                                id="granger-event-dropdown",
                                options=[{"label": event, "value": event} for event in event_types],
                                value=event_types[0],
                                multi=False,
                            ),
                            # Added slider for lag selection, enhancing interactivity
                            dcc.Slider(
                                id="lag-slider",
                                min=1,
                                max=2,  # Set to 2; adjust if max_granger_lag changes in analyze_event_types.py #TODO: better solution for this
                                step=1,
                                value=1,
                                marks={i: f"Lag {i}" for i in range(1, 3)},
                            ),
                            dcc.Graph(id="granger-bar-plot"),
                        ],
                    ),
                    # Tab 4: Correlation Heatmap - Improved for statistical clarity
                    dcc.Tab(
                        label="Correlation Heatmap",
                        children=[
                            html.H3("How All Storm Types Relate to Temperature"),
                            html.P(
                                "This heatmap shows the strength of the relationship between temperature changes and each storm type's impact. ",
                                "Darker colors indicate stronger correlations. Stars (*) mark statistically significant relationships (p < 0.05). ",
                                "Hover over cells for exact values.",
                            ),
                            dcc.Graph(id="correlation-heatmap"),
                        ],
                    ),
                ]
            ),
        ]
    )

    # Callback 1: Update Time Series Plot with descriptive labels
    @dash_callback(Output("time-series-plot", "figure"), [Input("event-dropdown", "value")])
    def update_time_series(event: str) -> go.Figure:
        event_data = merged_df[merged_df["event_type"] == event]
        # Improved labels for clarity: "Temperature Change (°C)" and "Storm Impact Index"
        fig = px.line(
            event_data,
            x="year",
            y=["temp_anomaly", "composite_index"],
            title=f"Trends for {event}",
            labels={
                "value": "Value",
                "variable": "Series",
                "year": "Year",
                "temp_anomaly": "Temperature Change (°C)",
                "composite_index": "Storm Impact Index",
            },
        )
        fig.update_layout(yaxis_title="Measure")
        return fig

    # Callback 2: Update Scatter Plot with simplified statistical annotation
    @dash_callback(Output("scatter-plot", "figure"), [Input("scatter-event-dropdown", "value")])
    def update_scatter_plot(event: str) -> go.Figure:
        event_data = merged_df[merged_df["event_type"] == event]
        corr_info = corr_df[corr_df["event_type"] == event].iloc[0]

        annotation_text = (
            f"Link strength: {corr_info['spearman_corr']:.2f} "
            f"(chance likelihood: {corr_info['spearman_p_perm']:.2f}), "
            f"range: [{corr_info['spearman_ci_lower']:.2f}, {corr_info['spearman_ci_upper']:.2f}], "
            f"data points: {corr_info['years_count']}"
        )

        fig = px.scatter(
            event_data,
            x="temp_anomaly",
            y="composite_index",
            trendline="ols",
            title=f"Temperature vs Storm Impact: {event}",
            labels={
                "temp_anomaly": "Temperature Anomaly (°C)",
                "composite_index": "Storm Impact Index",
            },
            hover_data=["year"],
        )
        fig.update_layout(
            annotations=[
                {
                    "text": annotation_text,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.05,
                    "y": 0.95,
                    "showarrow": False,
                    "font": {"size": 12},
                }
            ]
        )
        return fig

    # Callback 3: Update Granger Causality Bar Plot with forward and reverse causality
    @dash_callback(
        Output("granger-bar-plot", "figure"), [Input("granger-event-dropdown", "value"), Input("lag-slider", "value")]
    )
    def update_granger_bar(event: str, lag: int) -> go.Figure:
        event_corr = corr_df[corr_df["event_type"] == event]

        # Fixed extraction of p-values and added reverse causality
        forward_col = f"aicc_params_ftest_p_lag{lag}"
        reverse_col = f"aicc_rev_params_ftest_p_lag{lag}"
        n_rows = len(event_corr)

        try:
            p_values = event_corr.loc[:, [forward_col, reverse_col]].fillna(1.0).to_numpy()
            forward_p, reverse_p = p_values.T
        except KeyError:
            # If columns are missing, return arrays of 1.
            forward_p = np.ones(n_rows)
            reverse_p = np.ones(n_rows)

        # Bar plot now shows both directions of casuality
        fig = go.Figure(
            data=[
                go.Bar(name="Temp \u2192 Storm", x=[f"Lag {lag}"], y=[forward_p]),
                go.Bar(name="Storm \u2192 Temp", x=[f"Lag {lag}"], y=[reverse_p]),
            ]
        )
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            annotation_text="Significance Threshold (0.05)",
        )
        fig.update_layout(
            title=f"Casuality p-values for {event} at Lag {lag}",
            yaxis_title="p-value",
            yaxis_range=[0, 1],
            barmode="group",
        )
        return fig

    # Callback 4: Update Correlation Heatmap with significance markers
    @dash_callback(
        Output("correlation-heatmap", "figure"),
        [Input("event-dropdown", "value")],
    )
    def update_heatmap(_) -> go.Figure:  # type: ignore[no-untyped-def]
        # Chain pivot table creation with temp_anomaly addition
        pivot_df = merged_df.pivot_table(
            index="year", columns="event_type", values="composite_index", aggfunc="first"
        ).assign(temp_anomaly=merged_df.groupby("year")["temp_anomaly"].first())

        # TODO: ensure this is fine with sample sizes and lags etc.. check `analyze_event_types.py`
        # Calculate correlations and filter
        corr_with_temp = (
            pivot_df.corr(method="spearman")
            .loc[:, ["temp_anomaly"]]
            .drop("temp_anomaly")
            .sort_values(by="temp_anomaly", ascending=False)
        )

        # Optimize p-value handling and significance marking
        significance = corr_df.set_index("event_type")["spearman_p"].lt(0.05).replace({True: "*", False: ""})

        # Create text array
        text = (
            corr_with_temp.round(2)
            .astype(str)
            .add(significance.reindex(corr_with_temp.index, fill_value=""))
            .to_numpy()
            .reshape(-1, 1)
        )

        # Generate heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_with_temp.to_numpy(),
                x=["Temperature Change"],
                y=corr_with_temp.index,
                text=text,
                hoverinfo="text",
                colorscale="RdBu_r",
                zmin=1,
                zmax=1,
            )
        )
        fig.update_layout(
            title="Correlation between Temperature and Storm Types",
            xaxis_title="",
            yaxis_title="Storm Type",
        )
        return fig

    # Run the Dash app
    app.run(debug=True)
