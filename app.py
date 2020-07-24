import dash
import numpy as np
import pandas as pd
import base64
import os
import io
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
from data_processing_utils._processing_funcs import ResultProcessing
from dash.exceptions import PreventUpdate

from dash_extensions.callback import CallbackCache, DiskCache

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    prevent_initial_callbacks=True
)

app.title = 'LGP'

# Create (server side) disk cache.
cc = CallbackCache(cache=DiskCache(cache_dir="cache_dir"))

server = app.server

app.config['suppress_callback_exceptions'] = True

# -------------     layout code    ------------------
app.layout = html.Div(
    [
        # The memory store reverts to the default on every page refresh
        dcc.Loading(dcc.Store(id='filtered-result-store'), fullscreen=True),
        dcc.Loading(dcc.Store(id='raw-result-store'), fullscreen=True),
        dcc.Loading(dcc.Store(id='ori-df-store'), fullscreen=True),

        # --- website title ---
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("mib_logo.jpeg"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "SMILE: System Metabolomics using Interpretable Learning and Evolution",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Result Interpretation and Visualization", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Github", id="learn-more-button"),
                            href="https://github.com/ChengyuanSha/linear_genetic_programming",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        # --- File upload ---
        html.Div([
            dcc.Markdown(
                '''
                ##### Upload your data here. Your data will be removed after refreshing the page.
                '''),
            # pickle file upload
            html.Div([
                dcc.Upload(
                    id='upload-result-data',
                    children=html.Div([
                        html.H6('Upload pickle result file here'),
                        html.P('(This is the result file of running SMILE)'),
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    className="pretty_container six columns",
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
            ],
                className="row"
            ),
            # original data file upload
            html.Div([
                dcc.Upload(
                    id='upload-ori',
                    children=html.Div([
                        html.H6('Upload original dataset csv file here'),
                        html.P('(Please set the first column as the label, and the rests are metabolite features)'),
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    className="pretty_container six columns",
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
            ],
                className="row"
            ),
            html.Br(),
            # sample file link
            html.Div([
                html.A(
                    html.Button(
                        'Download sample pickle result data',
                        className='control-download'
                    ),
                    href=os.path.join('assets', 'sample_data', 'lgp_sample.pkl'),
                    download='lgp_sample.pkl',
                    className="four columns"
                ),

                html.A(
                    html.Button(
                        'Download sample csv ori data',
                        className='control-download'
                    ),
                    href=os.path.join('assets', 'sample_data', 'sample_alzheimer_vs_normal.csv'),
                    download='sample_alzheimer_vs_normal.csv',
                    className="four columns"
                ),

                html.Button(
                    "Refresh Result",
                    id="main-render-button",
                    className="three columns"
                )
            ],
                className="row"
            ),
            html.Br(),
        ],
        ),
        # main page
        html.Div(id='main-visualization-content')

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"}
)


def render_main_visualization_layout(available_indicators):
    return html.Div([
        # ---------- sliders/filters -----------
        html.Div([

            html.Div([
                html.H6('Choose testing accuracy threshold to filter models'),

                dcc.Slider(
                    id='testing-acc-filter-slider',
                    min=70,
                    max=100,
                    value=85,
                    marks={str(num): str(num) + '%' for num in range(70, 101, 5)},
                    step=5,
                    updatemode='drag'
                ),

            ],
                className="pretty_container six columns",
            ),

            html.Div(
                [html.H6(id="ori_model_count"), html.P("models generated by SMILE")],
                className="pretty_container three columns",
            ),
            html.Div(
                [html.H6(id="filtered_by_accuracy_text"),
                 html.P("models after filtering")],
                className="pretty_container three columns",
            ),

        ],
            className="row container-display",
        ),

        html.Br(),

        # ------------------   Feature Occurrence  --------------

        dcc.Tabs(children = [

            dcc.Tab(label='Feature Importance Analysis', children=[

                html.Div([
                    html.Div([
                        html.H6('Look at model with ____ effective features'),

                        dcc.RadioItems(id='prog-len-filter-slider', labelStyle={'display': 'inline-block'}),

                    ],
                        className="pretty_container six columns",
                    ),

                    html.Div(
                        [html.H6(id="filtered_by_len_text"),
                         html.P("models after selection")],
                        className="pretty_container four columns",
                    )
                    ],
                    className="row container-display"
                ),

                dcc.Tabs([
                    dcc.Tab(label='Feature Occurrence Analysis', children=[
                        html.Div([
                            html.Div([
                                dcc.Markdown(
                                    '''
                                    * Click a feature on **Feature Occurrence graph**. models containing that feature 
                                    will be shown in the **Model Accuracies graph**.  
                                    * Click on a point on **Model Accuracy graph**, the predictive model will be 
                                    shown in "Detailed Model Info".
                                    '''
                                ),

                                html.Br(),

                                dcc.Graph(
                                    id='filtered-occurrences-scatter',
                                )
                                ],
                                id="left-column",
                                className="pretty_container six columns",
                            ),
                            html.Div([
                                html.Div([
                                    dcc.Graph(
                                        id='filtered-accuracy-scatter',
                                    )
                                ],
                                    id="info-container",
                                    className="pretty_container",
                                ),

                                html.Div([
                                    dcc.Markdown("""
                                        **Detailed Model Info:**  
                                    """),

                                    html.Pre(id='model-click-data',
                                             style={
                                                 'border': 'thin lightgrey solid',
                                                 'overflowX': 'scroll'
                                             }),
                                ],
                                    id="accGraphContainer",
                                    className="pretty_container",
                                )
                                ],
                                className="six columns",
                            ),

                            ],
                            className="row flex-display",
                        ),
                    ]),

                    dcc.Tab(label='Pairwise Co-occurrence Analysis', children=[
                        # ---------------  Pairwise analysis -----------------
                        html.Div([
                            html.Div([
                                dcc.Markdown('''
                                    * Need to look at models with at least **two effective features**
                                    * Click on a cell in **Feature Pairwise Co-occurrence graph** to see 
                                    the scatter plot on the right side.   
                                '''),
                                ], className="pretty_container twelve columns"
                            ),
                            ],
                            className="row flex-display",
                            #style={'align-items': 'center', 'justify-content': 'center'},
                        ),

                        # Two Feature Comparision in the website
                        html.Div([
                            # Two Feature co-occurrence in website
                            html.Div([

                                html.Div(
                                    dcc.Graph(
                                        id='co-occurrence-graph'
                                    )
                                )

                            ],
                                id='left-column',
                                className="pretty_container four columns"
                            ), # end co-occurrence

                            # Two feature scatter plot, update based on x, y filter, see the callback
                            html.Div([
                                dcc.Markdown(
                                    '''
                                    * You can also manually choose X axis / Y axis
                                    '''
                                ),
                                # two manual filters
                                html.Div([
                                    dcc.Dropdown(
                                        id='crossfilter-xaxis-column',
                                        options=[{'label': str(i) + ': ' + str(n), 'value': i} for i, n in
                                                 available_indicators],
                                        value='0'
                                    ),
                                ],
                                ),

                                html.Div([
                                    dcc.Dropdown(
                                        id='crossfilter-yaxis-column',
                                        options=[{'label': str(i) + ': ' + str(n), 'value': i} for i, n in
                                                 available_indicators],
                                        value='1'
                                    ),
                                ],
                                ),

                                html.Br(),

                                dcc.Graph(
                                    id='scatter',
                                )
                                ], id='right-column',
                                className="pretty_container eight columns",
                            ),
                        ],
                            className="row flex-display"
                        ),
                    ]),
                ],
                vertical=True,
                parent_style={'flex-direction': 'column',
                              '-webkit-flex-direction': 'column',
                              '-ms-flex-direction': 'column',
                              'display': 'flex'},
                style={'width': '25%',
                       'float': 'left'},
                ), # end sub tabs
            ]
            ), # end tab 1


            dcc.Tab(label='Co-occurrence Network Analysis', children=[
                # ---------- network analysis ------------
                # network filter
                html.Div(
                    html.Div([
                        dcc.Markdown(
                            '''
                            ##### Network of Top ____% Most Common Metabolite Pairs  
                            '''
                        ),

                        dcc.Slider(
                            id='network-filter',
                            min=1,
                            max=10,
                            value=3,
                            marks={str(num): str(num) + '%' for num in range(1, 11, 1)},
                            step=1,
                            updatemode='drag'
                        ),

                        dcc.Markdown(
                            '''
                            * Top % most common metabolite pairs are represented as edges and their two end points 
                            * The pairwise co-occurrences are shown as edge weight
                            '''
                        )

                    ],
                        className="pretty_container eleven columns",
                    ),
                    className='container-display'
                ),

                # feature network
                html.Div(id='network'),
            ]), # end tab 2
            ],
            style={ 'font-size': '150%' },
        ), # end tabs
    ])


@cc.callback(
    Output('filtered-accuracy-scatter', 'figure'),
    [Input('filtered-occurrences-scatter', 'clickData'),
     Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data'),
     Input('ori-df-store', 'data')])
def update_accuracy_graph_based_on_clicks(clickData, prog_len, result_data, ori_df):
    names = ResultProcessing.read_dataset_names(ori_df)
    if clickData is not None:
        result_data.calculate_featureList_and_calcvariableList()
        feature_num = int(clickData['points'][0]['x'][1:])  # extract feature index data from click
        m_index = result_data.get_index_of_models_given_feature_and_length(feature_num, prog_len)
        testing_acc = [result_data.model_list[i].testingAccuracy for i in m_index]
        m_index = ['m' + str(i) for i in m_index]
        return {
            'data': [
                {'x': m_index,
                 'y': testing_acc,
                 'mode': 'markers',
                 'marker': {'size': 3}
                 },
            ],
            'layout': {
                'title': '<b>Model Accuracy</b>'+ '<br>' + 'Models containing feature ' + str(names[feature_num]) ,
                'xaxis': {'title': 'model index'},
                'yaxis': {'title': 'accuracy'},
                'clickmode': 'event+select'
            }
        }
    return {'layout':
            {'title': '<b>Model Accuracy</b>'}}


@cc.callback(
    [Output('filtered-occurrences-scatter', 'figure'),
     Output("filtered_by_accuracy_text", "children"),
     Output("filtered_by_len_text", "children")],
    [Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data'),
     Input('ori-df-store', 'data')])
def update_occurrence_graph(pro_len, result_data, ori_df):
    names = ResultProcessing.read_dataset_names(ori_df)
    result_data.calculate_featureList_and_calcvariableList()
    features, num_of_occurrences, cur_feature_num = result_data.get_occurrence_from_feature_list_given_length(pro_len)
    hover_text = [names[i] for i in features]
    features = ['f' + str(i) for i in features]
    return {
               'data': [{
                   'x': features,
                   'y': num_of_occurrences,
                   'type': 'bar',
                   'hoverinfo': 'text',
                   'text': hover_text
               }],
               'layout': {
                   'title': '<b>Feature Occurrences</b>',
                   'xaxis': {'title': 'feature index'},
                   'yaxis': {'title': 'occurrence'},
               },
           }, len(result_data.model_list), cur_feature_num


@cc.callback(
    Output('co-occurrence-graph', 'figure'),
    [Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data'),
     Input('ori-df-store', 'data')])
def update_co_occurrence_graph(pro_len, result_data, ori_df):
    names = ResultProcessing.read_dataset_names(ori_df)
    # result_data = jsonpickle.decode(result_data)
    result_data.calculate_featureList_and_calcvariableList()
    if pro_len == 'All' or pro_len > 1:
        cooc_matrix, feature_index = result_data.get_feature_co_occurences_matrix(pro_len)
        hover_text = []
        for yi, yy in enumerate(feature_index):
            hover_text.append([])
            for xi, xx in enumerate(feature_index):
                hover_text[-1].append(
                    'X: {}<br />Y: {}<br />Count: {}'.format(names[int(xx)], names[int(yy)], cooc_matrix[xi, yi]))
        feature_index = ['f' + str(i) for i in feature_index]
        return {
            'data': [{
                'z': cooc_matrix,
                'x': feature_index,
                'y': feature_index,
                'type': 'heatmap',
                'colorscale': 'Viridis',
                'hoverinfo': 'text',
                'text': hover_text
            }],
            'layout': {
                'title': '<b>Feature Pairwise Co-occurrence</b> ',
            }
        }
    return {
            'layout': {
                'title': 'Feature Pairwise Co-occurrence ',
            }
        }


@cc.callback(
    Output('scatter', 'figure'),
    [Input('prog-len-filter-slider', 'value'),
     Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('co-occurrence-graph', 'clickData'),
     Input('ori-df-store', 'data')])
def update_feature_comparision_graph_using_filters(trigger, xaxis_column_index, yaxis_column_index, co_click_data, ori_df):
    X, y = ResultProcessing.read_dataset_X_y(ori_df)
    names = ResultProcessing.read_dataset_names(ori_df)
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'crossfilter-xaxis-column' or trigger_id == 'crossfilter-yaxis-column':
        xaxis_column_index = int(xaxis_column_index)
        yaxis_column_index = int(yaxis_column_index)
    elif trigger_id == 'co-occurrence-graph':
        xaxis_column_index = int(co_click_data['points'][0]['x'][1:])
        yaxis_column_index = int(co_click_data['points'][0]['y'][1:])
    # type_name = ['AD', 'Normal']
    unique_label = ori_df['category'].unique()
    return {
        'data': [dict(
            x=X[:, int(xaxis_column_index)][y == type],
            y=X[:, int(yaxis_column_index)][y == type],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'},
            },
            name=type
        ) for type in unique_label
        ],
        'layout': dict(
            xaxis={
                'title': names[int(xaxis_column_index)],
                'type': 'linear'
            },
            yaxis={
                'title': names[int(yaxis_column_index)],
                'type': 'linear'
            },
            hovermode='closest',
            clickmode='event+select',
            title='<b>Two-Feature Scatter Plot</b>'
        )
    }

@cc.callback(
    Output('model-click-data', 'children'),
    [Input('filtered-accuracy-scatter', 'clickData'),
     Input('filtered-result-store', 'data'),
     Input('ori-df-store', 'data')])
def update_model_click_data(clickData, result_data, ori_df):
    names = ResultProcessing.read_dataset_names(ori_df)
    if clickData is not None:
        i = int(clickData['points'][0]['x'][1:])
        return result_data.convert_program_str_repr(result_data.model_list[i], names)


# ------   upload data section  ----------
def parse_contents_result(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'pkl' in filename:
            # initialize staff
            result = ResultProcessing()
            result.load_models_directly(io.BytesIO(decoded))
            return result
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ],
        )


@cc.cached_callback([Output('raw-result-store', 'data')],
                    [Input('upload-result-data', 'contents')],
                    [State('upload-result-data', 'filename')])
def update_file_output(contents, filename):
    # display read file status and update main visualization Div
    if contents is None:
        raise PreventUpdate
    global_result = parse_contents_result(contents[0], filename[0])
    return global_result


def parse_contents_ori(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ],
        )


@cc.cached_callback([Output('ori-df-store', 'data')],
                    [Input('upload-ori', 'contents')],
                    [State('upload-ori', 'filename')])
def update_file_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    ori_df = parse_contents_ori(contents[0], filename[0])
    return ori_df


# --- end upload data section ---

@cc.callback([Output('main-visualization-content', 'children')],
             [Input('main-render-button', 'n_clicks'),
              Input('ori-df-store', 'data')])
def update_main_page(n_clicks, ori_df):
    names = ResultProcessing.read_dataset_names(ori_df)
    index_list = [i for i in range(len(names))]
    available_indicators = list(zip(index_list, names))
    if n_clicks == 0:
        raise PreventUpdate
    else:
        return render_main_visualization_layout(available_indicators)

@cc.callback([Output('ori_model_count', 'children')],
             [Input('prog-len-filter-slider', 'value'),
              Input('raw-result-store', 'data')])
def display_ori_count(trigger, raw_result):
    return str(len(raw_result.model_list))


@cc.callback([Output('prog-len-filter-slider', 'options')],
             [Input('filtered-result-store', 'data')])
def set_prog_len_radiobutton_and_update_filtered_data(result_data):
    result_data.calculate_featureList_and_calcvariableList()
    length_list = sorted(list(set([len(i) for i in result_data.feature_list])))
    length_list = [i for i in length_list if i > 0]
    return [{'label': str(i), 'value': i} for i in length_list] + [{'label': 'All', 'value': 'All'}]


@cc.cached_callback([Output('filtered-result-store', 'data')],
                    [Input('testing-acc-filter-slider', 'value'),
                     Input('raw-result-store', 'data')])
def update_filtered_data(testing_acc, result_data):
    result_data.model_list = [i for i in result_data.model_list if
                              float(i.testingAccuracy) >= ((testing_acc) / 100)]
    result_data.calculate_featureList_and_calcvariableList()
    return result_data


@cc.callback(Output('prog-len-filter-slider', 'value'),
             [Input('prog-len-filter-slider', 'options')])
def set_prog_len_value(available_options):
    return available_options[0]['value']


@cc.callback(Output('network', 'children'),
             [Input('filtered-result-store', 'data'),
              Input('ori-df-store', 'data'),
              Input('network-filter', 'value')])
def create_network(result_data, ori_df, top_percentage):
    top_percentage = top_percentage * 0.01
    names = ResultProcessing.read_dataset_names(ori_df)
    df = result_data.get_network_data(names, top_percentage)
    # error catching, when no data available
    if df.empty:
        return html.Div(
            html.Div(
                dcc.Markdown(
                    '''
                    ##### No network graph in given selection, try to decrease testing accuracy filter.
                    '''),
                className='pretty_container eleven columns',
            ),
            className='container-display',
        )
    nodes = [
        {
            'data': {'id': node, 'label': node},
            'position': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)}
        }
        for node in np.unique(df[['f1', 'f2']].values)
    ]
    edges = [
        {'data': {'source': df['f1'][index], 'target': df['f2'][index], 'weight': df['weight'][index]}}
        for index, row in df.iterrows()
    ]
    elements = nodes + edges
    return html.Div(
        html.Div([
            cyto.Cytoscape(
                id='cytoscape-layout-1',
                elements=elements,
                style={'width': '100%', 'height': '350px'},
                layout={
                    'name': 'circle'
                },
                zoomingEnabled=False,
                stylesheet=[
                    {
                        'selector': 'node',
                        'style': {
                            'label': 'data(label)'
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'label': 'data(weight)'
                        }
                    },
                    {
                        'selector': '[weight > 0]',
                        'style': {
                            'line-color': '#CCCCCC'
                        }
                    }
                ]
            ) # end cytoscape
        ],
            className='pretty_container eleven columns',
        ),
        className='container-display',
    )


cc.register(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True)