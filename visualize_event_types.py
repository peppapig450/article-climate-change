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
                            html.H3("Correlation Between Temperature and Event Index"),
                            dcc.Dropdown(
                                id="scatter-event-dropdown",
                                options=[
                                    {"label": event, "value": event}
                                    for event in event_types
                                ],
                                value=event_types[0],
                                multi=False,
                            ),
                            dcc.Graph(id="scatter-splot"),
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
