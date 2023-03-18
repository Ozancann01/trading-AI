import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests
import json
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def run_dashboard(trading_data):
    app.run_server(host='0.0.0.0', port=8050)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1("AI Trading Dashboard"),
                    className="mb-2 mt-2"
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="balance-chart"),
                    className="mb-4",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H2("Trading Actions"),
                            dcc.Interval(
                                id="interval",
                                interval=1 * 1000,  # in milliseconds
                                n_intervals=0
                            ),
                            html.Div(id="trading-actions"),
                        ]
                    ),
                    className="mb-4",
                ),
            ]
        ),
    ],
    fluid=True,
)

def get_trading_data():
    try:
        with open('trading_data.json', 'r') as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()



@app.callback(Output("trading-actions", "children"), [Input("interval", "n_intervals")])
def update_trading_actions(_):
    df = get_trading_data()
    if df.empty:
        return html.P("No trading data available.")
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

@app.callback(Output("balance-chart", "figure"), [Input("interval", "n_intervals")])
def update_balance_chart(_):
    df = get_trading_data()
    if df.empty:
        return px.line(title="Balance Over Time")
    fig = px.line(df, x="timestamp", y="balance", title="Balance Over Time")
    return fig

