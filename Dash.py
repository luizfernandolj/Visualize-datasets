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
mid_size_test = str((df['MaxSizeTest'].min() + df['MaxSizeTest'].max())/2)

app.layout = dbc.Container([
    navbar := dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.NavItem(dbc.NavLink("Page 2", href="#")),
            dbc.NavItem(dbc.NavLink("Page 3", href="#"))
        ], style={"height":"50px"}, brand="NavbarSimple", brand_href="#", color="gray", dark=True,
    ),

    dcc.Markdown("Datasets for experiments", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            dbc.Col([
                dbc.Row(html.Div("Sizes"), style={'text-align': 'center'}),
                dbc.Row(html.Div("Alphas"), style={'text-align': 'center'}),
                dbc.Row(html.Div("Complexity (AUC)"), style={'text-align': 'center'}),
                dbc.Row(html.Div("Max Size Test"), style={'text-align': 'center'}),
            ],width=3, style={ 'display': 'flex', 'flex-direction': 'column','height':'300px', 'justify-content':'space-between'}, className='mt-3-mb-4'),

            dbc.Col([
                dbc.Row([
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
                ]),

                dbc.Row([
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
                ]),

                dbc.Row([
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
                ]),
                dbc.Row([
                    max_size_test_range := dcc.RangeSlider(df['MaxSizeTest'].min(), df['MaxSizeTest'].max(), 100,
                                                     value=[df['MaxSizeTest'].min(), df['MaxSizeTest'].max()],
                                                     marks={
                                                         str(df['MaxSizeTest'].min()): {'label': str(df['MaxSizeTest'].min()),
                                                                           'style': {'color': 'white'}},
                                                         mid_size_test: {'label': mid_size_test, 'style': {'color': 'white'}},
                                                         str(df['MaxSizeTest'].max()): {'label': str(df['MaxSizeTest'].max()),
                                                                           'style': {'color': 'white'}}
                                                     },
                                                     tooltip={"placement": "bottom", "always_visible": True},
                                                     id='max_size_test_range')
                ]),
            ],width=6, style={'display':'flex', 'flex-direction':'column', 'height':'300px', 'justify-content':'space-between'},className='mt-3-mb-4')
        ], style={'display': 'flex', 'flex-direction': 'row', 'width':'50%', 'text-align':'center'}),

    html.Div([
        my_table := dash_table.DataTable(data=df.to_dict('records'),
                                         id='my_table',
                                         columns=[{"name": i, "id": i} for i in df.columns],
                                         style_table={'height': '300px', 'overflowY': 'auto'},
                                         fixed_rows={'headers': True},
                                         style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                       'fontWeight': 'bold', 'font-size': '12px',
                                                       'text-align':'center'},
                                         page_action='none',
                                         style_data={'height': '25px',
                                                     'margin': '10px',
                                                     'minWidth': '60px',
                                                     'maxWidth': '120px',
                                                     'font-size':'10px',
                                                     'color': 'white',
                                                     'backgroundColor': 'rgb(50, 50, 50)'}),
            ], style={"width":"50%"}),
    ], style={'display':'flex', 'flex-direction': 'row', 'margin': '0px 40px'}),
    dcc.Markdown(children="", id='number_rows', style={"textAlign": "center", 'padding-top':'30px'}),
], fluid=True)


@callback(
    Output(component_id='my_table', component_property='data'),
    Output(component_id='number_rows', component_property='children'),
    Input(component_id='size_range', component_property='value'),
    Input(component_id='alpha_range', component_property='value'),
    Input(component_id='complex_range', component_property='value'),
    Input(component_id='max_size_test_range', component_property='value')
)
def update_output(size_r, alpha_r, complex_r, max_size):
    dff = df.copy()
    dff = dff[(dff['size'] >= size_r[0]) & (dff['size'] <= size_r[1])]
    dff = dff[(dff['%Positive'] >= alpha_r[0]) & (dff['%Positive'] <= alpha_r[1])]
    dff = dff[(dff['AUC'] >= complex_r[0]) & (dff['AUC'] <= complex_r[1])]
    dff = dff[(dff['MaxSizeTest'] >= max_size[0]) & (dff['MaxSizeTest'] <= max_size[1])]

    return dff.to_dict('records'), f"Number of rows: {len(dff.index)}"

if __name__ == "__main__":
    app.run_server(debug=True, port=3004)

