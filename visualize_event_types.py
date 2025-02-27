import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


def launch_dashboard(merged_df: pd.DataFrame, corr_df: pd.DataFrame):
    """Launch an interactive Dash dashboard with multiple visualizations."""
    app = dash.Dash(__name__)

    # Unique event types for dropdowns
    event_types = merged_df["event_type"].unique()

    # Define the dashboard layout
    app.layout = html.Div(
        [
            html.H1("Climate and Storm Event Analysis Dashboard"),
            dcc.Tabs(
                [
                    # Tab 1: Time Series (Simpler Visual)
                    dcc.Tab(
                        label="Time Series",
                        children=[
                            html.H3(
                                "Explore Temperature Anomalies and Event Indices over Time"
                            ),
                            dcc.Dropdown(
                                id="event-dropdown",
                                options=[
                                    {"label": event, "value": event}
                                    for event in event_types
                                ],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="time-series-plot"),
                        ],
                    ),
                    # Tab 2: Correlation Scatter (Simpler Visual with Stats)
                    dcc.Tab(
                        label="Correlation Scatter",
                        children=[
                            html.H3("Correlation Between Temperature Anomaly and Composite Index"),
                            dcc.Dropdown(
                                id="scatter-event-dropdown",
                                options=[
                                    {"label": event, "value": event}
                                    for event in merged_df["event_type"].unique()
                                ],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="scatter-plot"),
                        ],
                    ),
                    # Tab 3: Granger Causality (Detailed Statistical Plot)
                    dcc.Tab(
                        label="Granger Causality",
                        children=[
                            html.H3("Granger Causality p-values Across Lags"),
                            dcc.Dropdown(
                                id="granger-event-dropdown",
                                options=[
                                    {"label": event, "value": event}
                                    for event in event_types
                                ],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="granger-bar-plot"),
                        ],
                    ),
                    # Tab 4: Correlation Heatmap (Detailed Statistical Plot)
                    dcc.Tab(
                        label="Correlation Heatmap",
                        children=[
                            html.H3("Spearman Correlation Across All Event Types"),
                            dcc.Graph(id="correlation-heatmap"),
                        ],
                    ),
                ]
            ),
        ]
    )

    # Callback 1: Update Time Series Plot
    @app.callback(
        Output("time-series-plot", "figure"), [Input("event-dropdown", "value")]
    )
    def update_time_series(event):
        event_data = merged_df[merged_df["event_type"] == event]
        fig = px.line(
            event_data,
            x="year",
            y=["temp_anomaly", "composite_index"],
            title=f"Time Series: {event}",
            labels={"value": "Value", "variable": "Series"},
        )
        fig.update_layout(yaxis_title="Value")
        return fig

    # Callback 2: Update Scatter Plot
    @app.callback(
        Output("scatter-plot", "figure"), [Input("scatter-event-dropdown", "value")]
    )
    def update_scatter_plot(event):
        event_data = merged_df[merged_df["event_type"] == event]
        corr_info = corr_df[corr_df["event_type"] == event].iloc[0]
        
        annotation_text = (
            f"Composite: {corr_info['spearman_corr']:.3f} (p={corr_info['spearm_p_perm']:.3f}), "
            f"CI=[{corr_info['spearman_ci_lower']:.3f}, {corr_info['spearman_ci_upper']:.3f}], "
            f"n={corr_info['years_count']}"
        )
        
        fig = px.scatter(
            event_data,
            x="temp_anomaly",
            y="composite_index",
            trendline="ols",
            title=f"Scatter Plot: {event}",
            labels={
                "temp_anomaly": "Temperature Anomaly",
                "composite_index": "Composite Index",
            },
        )
        fig.update_layout(
            annotations=[dict(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.05,
                y=0.95,
                showarrow=False,
                font=dict(size=12)
            )]
        )
        return fig

    # TODO: doesnt look right
    # Callback 3: Update Granger Causality Bar Plot
    @app.callback(
        Output("granger-bar-plot", "figure"), [Input("granger-event-dropdown", "value")]
    )
    def update_granger_bar(event):
        event_corr = corr_df[corr_df["event_type"] == event]
        lags = range(1, 3)  # Adjust based on max_granger_lag; here, max=2
        p_values = [
            event_corr[f"aicc_params_ftest_p_lag{lag}"].values[0]
            if f"aicc_params_ftest_p_lag{lag}" in event_corr.columns
            else 1.0
            for lag in lags
        ]
        fig = go.Figure([go.Bar(x=[f"Lag {lag}" for lag in lags], y=p_values)])
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            annotation_text="Significance Threshold (0.05)",
        )
        fig.update_layout(
            title=f"Granger Causality p-values: {event}",
            yaxis_title="p-value",
            yaxis_range=[0, 1],
        )
        return fig

    # TODO: doesnt look right
    # Callback 4: Update Correlation Heatmap
    @app.callback(
        Output("correlation-heatmap", "figure"),
        [
            Input("event-dropdown", "value")
        ],  # Static for now, using event-dropdown as a placeholder
    )
    def update_heatmap(_):
        pivot_df = merged_df.pivot(
            index="year", columns="event_type", values="composite_index"
        )
        pivot_df["temp_anomaly"] = merged_df.groupby("year")["temp_anomaly"].first()
        corr_matrix = pivot_df.corr(method="spearman")
        fig = px.imshow(
            corr_matrix[["temp_anomaly"]]
            .drop("temp_anomaly")
            .sort_values(by="temp_anomaly", ascending=False),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Spearman Correlation: Temperature Anomaly vs. Event Types",
            labels={"color": "Correlation"},
        )
        return fig

    # Run the Dash app
    app.run_server(debug=True)

    app.run(debug=True)
