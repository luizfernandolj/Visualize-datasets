import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc

df = pd.read_csv("table.csv")

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

mid_size = str((df['size'].min() + df['size'].max()) / 2)
mid_alpha = str((df['%Positive'].min() + df['%Positive'].max()) / 2)
mid_auc = str((df['AUC'].min() + df['AUC'].max()) / 2)

app.layout = dbc.Container([
    dcc.Markdown("Datasets for experiments", style={"textAlign": "center"}),

    dbc.Label("Show number of rows"),

    html.Center([
        my_table := dash_table.DataTable(data=df.to_dict('records'),
                                             id='my_table',
                                             columns=[{"name": i, "id": i} for i in df.columns],
                                             style_table={'height': '500px', 'width':'60%','overflowY': 'auto'},
                                             fixed_rows={'headers': True},
                                             style_header={'backgroundColor': 'rgb(30, 30, 30)', 'fontWeight': 'bold', 'font-size':'18px'},
                                             page_action='none',
                                             style_data={'height': '35px',
                                                         'margin': '10px',
                                                         'minWidth': '50px',
                                                         'maxWidth': '150px',
                                                         'color': 'white',
                                                         'backgroundColor': 'rgb(50, 50, 50)'}),
    ]),

    html.Div([
        dbc.Row([
            dbc.Col(html.Div("Sizes"), style={'text-align': 'center', 'padding-bottom-top': '10px'}, width=3),
            dbc.Col(html.Div("Alphas"), style={'text-align': 'center', 'padding-bottom-top': '10px'}, width=3),
            dbc.Col(html.Div("Complexity (AUC)"), style={'text-align': 'center', 'padding-bottom-top': '10px'},
                    width=3),
        ], justify='between', className='mt-3-mb-4'),

        dbc.Row([
            dbc.Col([
                size_range := dcc.RangeSlider(df['size'].min(),
                                              df['size'].max(),
                                              500,
                                              value=[df['size'].min(), df['size'].max()],
                                              marks={
                                                  str(df['size'].min()): {'label': str(df['size'].min()),
                                                                     'style': {'color': 'white'}},
                                                  mid_size: {'label': mid_size, 'style': {'color': 'white'}},
                                                  str(df['size'].max()): {'label': str(df['size'].max()),
                                                                     'style': {'color': 'white'}
                                                                     }},
                                              tooltip={"placement": "bottom", "always_visible": True},
                                              id='size_range')
            ], width=3),

            dbc.Col([
                alpha_range := dcc.RangeSlider(df['%Positive'].min(),
                                               df['%Positive'].max(),
                                               0.01,
                                               value=[df['%Positive'].min(), df['%Positive'].max()],
                                               marks={
                                                   str(df['%Positive'].min()): {'label': str(df['%Positive'].min()),
                                                                           'style': {'color': 'white'}},
                                                   mid_alpha: {'label': mid_alpha, 'style': {'color': 'white'}},
                                                   str(df['%Positive'].max()): {'label': str(df['%Positive'].max()),
                                                                           'style': {'color': 'white'}
                                                                           }},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               id='alpha_range')
            ], width=3),

            dbc.Col([
                complex_range := dcc.RangeSlider(df['AUC'].min(), df['AUC'].max(), 0.01,
                                                 value=[df['AUC'].min(), df['AUC'].max()],
                                                 marks={
                                                     str(df['AUC'].min()): {'label': str(df['AUC'].min()),
                                                                       'style': {'color': 'white'}},
                                                     mid_auc: {'label': mid_auc, 'style': {'color': 'white'}},
                                                     str(df['AUC'].max()): {'label': str(df['AUC'].max()),
                                                                       'style': {'color': 'white'}}
                                                 },
                                                 tooltip={"placement": "bottom", "always_visible": True},
                                                 id='complex_range')
            ], width=3),
        ], justify="between", className='mt-3-mb-4')
    ], style={'display': 'flex', 'flex-direction': 'column', 'margin-top':'50px'})
], fluid=True)


@callback(
    Output(component_id='my_table', component_property='data'),
    Input(component_id='size_range', component_property='value'),
    Input(component_id='alpha_range', component_property='value'),
    Input(component_id='complex_range', component_property='value')
)
def update_output(size_r, alpha_r, complex_r):
    dff = df.copy()
    if size_r:
        dff = dff[(dff['size'] >= size_r[0]) & (dff['size'] <= size_r[1])]
    if alpha_r:
        dff = dff[(dff['Positives'] >= alpha_r[0]) & (dff['Positives'] <= alpha_r[1])]
    if complex_r:
        dff = dff[(dff['AUC'] >= complex_r[0]) & (dff['AUC'] <= complex_r[1])]

    return dff.to_dict('records')

if __name__ == "__main__":
    app.run_server(debug=True, port=3004)

