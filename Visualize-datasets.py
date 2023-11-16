import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc

df = pd.read_csv('table.csv')
df_exp = pd.read_csv('experiments\\experiments.csv')







app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])








##################################           STYLE           #################################


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "color":"#F5F5F5",
    "background-color": "#2e2f2e",
    "box-shadow":"rgba(0, 0, 0, 0.35) 0px 5px 15px"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "0",
    "top":0,
    "padding": "0 0",
    "background-color": "#F5F5F5",
    "height":"300vw",
}



##################################           SIDEBAR           #################################


sidebar = html.Div(
    [
        html.H2("Vizualize Datasets", className="display-4 text-center", style={"color":"#F5F5F5"}),
        html.Hr(),
        html.P(
            "Select the options to be shown on the graphs", className="lead text-center"
        ),
        dbc.Nav([
            html.Label('Size'),
            dcc.RangeSlider(
                        id='size-slider',
                        min=df['size'].min(),
                        max=df['size'].max(),
                        marks={i: {'label': str(i),'style': {'color': '#AAA'}} for i in range(df['size'].min(), df['size'].max() + 1, 150000)},
                        step=1,
                        value=[df['size'].min(), df['size'].max()],
                        tooltip={"placement": "bottom", "always_visible": True},
            ),

            html.Label('MaxSizeTest', className="pt-5"),
            dcc.RangeSlider(
                        id='max-test-size-slider',
                        min=df['MaxSizeTest'].min(),
                        max=df['MaxSizeTest'].max(),
                        marks={i: {'label': str(i),'style': {'color': '#AAA'}} for i in range(df['MaxSizeTest'].min(), df['MaxSizeTest'].max() + 1, 5000)},
                        step=500,
                        value=[df['MaxSizeTest'].min(), df['MaxSizeTest'].max()],
                        tooltip={"placement": "bottom", "always_visible": True},
            ),

            html.Label('Alpha', className="pt-5"),
            dcc.RangeSlider(
                        id='positive-cases-slider',
                        min=df['%Positive'].min(),
                        max=df['%Positive'].max(),
                        marks={i: {'label': str(i),'style': {'color': '#AAA'}} for i in np.round(np.arange(df['%Positive'].min(), df['%Positive'].max() + 1, 0.05), 2)},
                        step=0.01,
                        value=[df['%Positive'].min(), df['%Positive'].max()],
                        tooltip={"placement": "bottom", "always_visible": True},
            ),

            html.Label('Complexity (AUC)', className="pt-5"),
            dcc.RangeSlider(
                        id='auc-slider',
                        min=df['AUC'].min(),
                        max=df['AUC'].max(),
                        marks={i: {'label': str(i),'style': {'color': '#AAA'}} for i in np.round(np.arange(df['AUC'].min(), df['AUC'].max() + 1, 0.1), 2)},
                        step=0.01,
                        value=[df['AUC'].min(), df['AUC'].max()],
                        tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        vertical=True,
        pills=True),
        html.P(
                "", className="lead text-center pt-5 fw-bold fs-1", id="dataset_rows"
            ),
    ],
    style=SIDEBAR_STYLE,
)




##################################           CONTENT           #################################




content = html.Div([
    dash_table.DataTable(
                id='datatable',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                sort_action='native',
                sort_mode="multi",
                fixed_rows={'headers': True},
                style_as_list_view=True,
                style_table={'height': '500px', 'overflowY': 'auto', "box-shadow": "rgba(100, 100, 111, 0.2) 0px 7px 29px 0px"},
                style_cell={'height':'50px', 'text-align':'center','minWidth': '60px', 'maxWidth': '120px', 'color':'black', 'background-color':"#F4F4F4"},
                style_header={'background-color':"#2e2f2e", 'height':'50px', 'color':'white', 'font-size':'15px'}
    ),
    dbc.Row([
        dbc.Col([
            html.H2("Applying Quantifiers to datasets", className="display-6 text-center m-5", style={"color":"#2e2f2e"}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("We applied 11 quantifiers algorithms to each of the datasets, and we made severals experiments "
                "varying size-test, alpha, threshold and analyze the data with the absolute error", className="lead text-center m-5 text-md-start"
            ),
        ], width=6, style={"border":"1px solid #2e2f2e",
                           "background-color":"#2e2f2e",
                           "color":"#F5F5F5",
                           "box-shadow": "rgba(100, 100, 111, 0.2) 0px 7px 29px 0px",
                           "border-radius":"40px"})
    ], justify="center"),

    dbc.Row([
        dbc.Col([
            html.Label('Quantifiers', className="pt-5"),
            dcc.Dropdown(df_exp["quantifier"].unique(), 'All', id='qtf-dropdown'),
        ], width={"size":2, "offset":1}),

        dbc.Col([
            html.Label('Alpha', className="pt-5"),
            dcc.Slider(0, 20, 5, value=10,id='alpha-slider')
        ], width={"size":3}),

        dbc.Col([
            html.Label('Batch-sizes', className="pt-5"),
            dcc.Slider(0, 20, 5, value=10,id='batch-slider')
        ], width={"size":3}),

        dbc.Col([
            html.Label('Threshold', className="pt-5"),
            dcc.Slider(0, 20, 5, value=10,id='thr-slider')
        ], width={"size":2}),
    ], justify="around"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="line-ae")
        ])
    ])

],id="page-content", style=CONTENT_STYLE)



##################################           APP          #################################



app.layout = html.Div([
    sidebar,
    content,
])





##################################           CALLBACKS           #################################



@callback([
        Output('datatable', 'data'),
        Output('dataset_rows', "children"),
        Output('line-ae', 'figure'),
        Input('size-slider', 'value'),
        Input('positive-cases-slider', 'value'),
        Input('max-test-size-slider', 'value'),
        Input('auc-slider', 'value'),
    ]
)

def update_table(selected_size, selected_positive_cases, selected_max_test_size, selected_auc):
    filtered_df = df[
        (df['size'] >= selected_size[0]) & (df['size'] <= selected_size[1]) &
        (df['%Positive'] >= selected_positive_cases[0]) & (df['%Positive'] <= selected_positive_cases[1]) &
        (df['MaxSizeTest'] >= selected_max_test_size[0]) & (df['MaxSizeTest'] <= selected_max_test_size[1]) &
        (df['AUC'] >= selected_auc[0]) & (df['AUC'] <= selected_auc[1])
    ]
    figure=px.line(filtered_df, x="size", y="%Positive", title="Quantifiers absolute error", height=800)

    return filtered_df.to_dict('records'), f"{len(filtered_df.index)} Datasets", figure




##################################           RUN           #################################



if __name__ == "__main__":
    app.run_server(debug=True, port=8888)