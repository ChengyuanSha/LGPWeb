import dash
import numpy as np
import pandas as pd
import base64
import datetime
import pickle
import os
import copy
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
cc = CallbackCache(cache=DiskCache(cache_dir="cache_dir"), expire_after=10)

server = app.server

# need to suppress bc using dynamic tabs
app.config['suppress_callback_exceptions'] = True

# -------------     layout code    ------------------
app.layout = html.Div(
    [
        # The memory store reverts to the default on every page refresh
        dcc.Loading(dcc.Store(id='filtered-result-store')),
        dcc.Loading(dcc.Store(id='raw-result-store')),
        dcc.Loading(dcc.Store(id='ori-data-store')),

        # --- headline ---
        # empty Div to trigger javascript file for graph resizing
        #html.Div(id='signal', style={'display': 'none'}),

        # --- website title ---
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("gene.png"),
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
                                    "Linear Genetic Programming",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Result Visualization", style={"margin-top": "0px"}
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

            dcc.Upload(
                id='upload-result-data',
                children=html.Div([
                    html.H6('Upload pickle result file here'),
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="pretty_container six columns",
                # Allow multiple files to be uploaded
                multiple=True
            ),

            dcc.Upload(
                id='upload-ori',
                children=html.Div([
                    html.H6('Upload original dataset csv file here'),
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                className="pretty_container six columns",
                # Allow multiple files to be uploaded
                multiple=True
            ),

            html.A(
                html.Button(
                    'Download sample pickle result data',
                    className='control-download'
                ),
                href=os.path.join('assets', 'sample_result_data', 'lgp_sample.pkl'),
                download='lgp_sample.pkl'
            ),

            html.A(
                html.Button(
                    'Download sample csv ori data',
                    className='control-download'
                ),
                href=os.path.join('assets', 'sample_ori_data', 'sample_alzheimer_vs_normal.csv'),
                download='lgp_sample.pkl'
            ),

            html.Button(
                "Visualize Result",
                id="main-render-button",
            )
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
                html.H6('Choose testing set accuracy threshold to filter'),

                dcc.Slider(
                    id='testing-acc-filter-slider',
                    min=70,
                    max=100,
                    value=70,
                    marks={str(num): str(num) + '%' for num in range(70, 101, 5)},
                    step=5,
                    updatemode='drag'
                ),
                html.Div(id='updatemode-output-testing-acc', style={'margin-top': 20})

            ],
                className="pretty_container six columns",
            ),

            html.Div([
                html.H6('Choose number of Feature in a Program'),

                # dcc.Slider(id='prog-len-filter-slider',marks={int(each_len): 'Len' + str(each_len) for each_len in [1,2]}),
                dcc.RadioItems(id='prog-len-filter-slider', labelStyle={'display': 'inline-block'}),
                # html.Div([dcc.Slider(id='slider')], id='slider-keeper'),  # dummy slider
                # html.Div(id='prog-len-filter-slider'),
                html.Div(id='updatemode-output-proglenfilter', style={'margin-top': 20})
            ],
                className="pretty_container six columns",
            ),

        ],
            style={'align-items': 'center', 'justify-content': 'center'},
            className="row container-display",
        ),

        html.Div(
            [
                html.Div(
                    #[html.H6(str(len(original_result.model_list))), html.P("Original Model Count")],
                    #[html.H6(str(len(original_result.model_list))), html.P("Original Model Count")],
                    [html.H6(id="ori_model_count"), html.P("Original Model Count")],
                    className="mini_container",
                ),
                html.Div(
                    [html.H6(id="filtered_by_accuracy_text"), html.P("Model Count After Filtered by Testing Set Accuracy")],
                    className="mini_container",
                ),
                html.Div(
                    [html.H6(id="filtered_by_len_text"), html.P("Model Count After Filtered by Accuracy and Number of Feature")],
                    className="mini_container",
                ),
            ],
            className="row container-display",
        ),
        # ------------------   first 2 graphs in website  --------------
        html.Div([
            html.Div([
                dcc.Graph(
                    id='filtered-occurrences-scatter',
                )
            ],
                id="left-column",
                className="pretty_container six columns",
            ),

            html.Div([
                dcc.Graph(
                    id='filtered-accuracy-scatter',
                )
            ],
                id="right-column",
                className="pretty_container six columns",
            ),
        ],
            className="row flex-display",
        ),
        #     ------------------   model visualization  --------------
        html.Div([
            html.Div([
                dcc.Markdown("""
                    **Click Models On Model Accuracy Scatter Plot**  

                    Detailed Model Info:
                """),

                html.Pre(id='model-click-data',
                         style={
                             'border': 'thin lightgrey solid',
                             'overflowX': 'scroll'
                         }),
            ],
            className="pretty_container six columns",)
        ],
        className="row flex-display",
        ),

        # row 2 selectors in website
        html.Div([
            html.Div([
                dcc.Markdown('''
                            **Click on co-occurrence heat map to see two feature distribution in original data.**    
                            Or manually choose X axis / Y axis for two distribution graph on dropdown manual.
                        '''),

                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-xaxis-column',
                        options=[{'label': str(i) + ': ' + str(n), 'value': i} for i, n in available_indicators],
                        value='0'
                    ),
                ], style={'width': '49%', 'display': 'inline-block'},

                ),

                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-yaxis-column',
                        options=[{'label': str(i) + ': ' + str(n), 'value': i} for i, n in available_indicators],
                        value='1'
                    ),
                ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'},
                )
            ],
                className="pretty_container twelve columns"
            ),
        ],
            className="row flex-display",
            style={'align-items': 'center', 'justify-content': 'center'},
        ),

        # 4 row Two Feature Comparision in the website
        html.Div([
            # Two Feature Co-occurrence in website
            html.Div([
                html.Div(
                    html.H6('Two Feature Co-occurrence Analysis (Only For 2+ Features)')
                ),

                html.Div(
                    dcc.Graph(
                        id='co-occurrence-graph'
                    )
                )

            ],
                className="pretty_container four columns"
            ),

            # 2 feature scatter plot, update based on x, y filter, see the callback
            html.Div([
                dcc.Graph(
                    id='crossfilter-indicator-scatter',
                    # hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ],
                className="pretty_container seven columns",
            ),
        ],
        className="row flex-display"
        ),

        # --- network analysis ----
        html.Div(id='network'),


    ])

@cc.callback(
    Output('filtered-accuracy-scatter', 'figure'),
    [Input('filtered-occurrences-scatter', 'clickData'),
     Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data')])
def update_accuracy_graph_based_on_clicks(clickData, prog_len, result_data):
    if clickData is not None:
        # result_data = jsonpickle.decode(result_data)
        result_data.calculate_featureList_and_calcvariableList()
        feature_num = int(clickData['points'][0]['x'][1:]) # extract data from click
        m_index = result_data.get_index_of_models_given_feature_and_length(feature_num, prog_len)
        testing_acc = [result_data.model_list[i].testingAccuracy for i in m_index]
        m_index = ['m' + str(i) for i in m_index]
        return {
                'data': [
                    {'x': m_index,
                     'y': testing_acc,
                     'mode':'markers',
                     'marker': {'size': 3}
                    },
                ],
                'layout': {
                    'title': 'Model accuracy containing feature ' + str(feature_num) + ' with' + str(prog_len) + ' features',
                    'xaxis': {'title': 'Program feature index'},
                    'yaxis': {'title': 'Num of occurrences'},
                    'clickmode': 'event+select'
                }
        }
    return {  'layout': {
                    'title': 'No graph in given selection. Click on Occurrence graph.'
                }}


@cc.callback(
    [Output('filtered-occurrences-scatter', 'figure'),
     Output("filtered_by_accuracy_text", "children"),
     Output("filtered_by_len_text", "children")],
    [Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data'),
     Input('ori-data-store', 'data')])
def update_occurrence_graph(pro_len, result_data, ori_data):
    names = ResultProcessing.read_dataset_names(ori_data)
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
                   'title': 'Occurrences of Features of ' + str(pro_len) + ' Feature Models',
                   'xaxis': {'title': 'Program feature index'},
                   'yaxis': {'title': 'Num of occurrences'},
               },
           }, len(result_data.model_list), cur_feature_num



@cc.callback(
    Output('co-occurrence-graph', 'figure'),
    [Input('prog-len-filter-slider', 'value'),
     Input('filtered-result-store', 'data'),
     Input('ori-data-store', 'data')])
def update_co_occurrence_graph(pro_len, result_data, ori_data):
    names = ResultProcessing.read_dataset_names(ori_data)
    #result_data = jsonpickle.decode(result_data)
    result_data.calculate_featureList_and_calcvariableList()
    if pro_len > 1:
        cooc_matrix, feature_index = result_data.get_feature_co_occurences_matrix(pro_len)
        hover_text = []
        for yi, yy in enumerate(feature_index):
            hover_text.append([])
            for xi, xx in enumerate(feature_index):
                hover_text[-1].append('X: {}<br />Y: {}<br />Count: {}'.format(names[int(xx)], names[int(yy)], cooc_matrix[xi, yi]))
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
                'title': 'Co-occurrence of ' + str(pro_len) + ' Feature Models',
                #'margin': dict(l=20, r=20, t=20, b=20)
            }
        }
    return {}


@cc.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('co-occurrence-graph', 'clickData'),
     Input('ori-data-store', 'data')])
def update_feature_comparision_graph_using_filters(xaxis_column_index, yaxis_column_index, co_click_data, ori_data):
    X, y  = ResultProcessing.read_dataset_X_y(ori_data)
    names = ResultProcessing.read_dataset_names(ori_data)
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'crossfilter-xaxis-column' or trigger_id == 'crossfilter-yaxis-column':
        xaxis_column_index = int(xaxis_column_index)
        yaxis_column_index = int(yaxis_column_index)
    elif trigger_id == 'co-occurrence-graph':
        xaxis_column_index = int(co_click_data['points'][0]['x'][1:])
        yaxis_column_index = int(co_click_data['points'][0]['y'][1:])
    # type_name = ['AD', 'Normal']
    unique_label = ori_data['category'].unique()
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
            title='Two Feature Comparision'
        )
    }

@cc.callback(
    Output('model-click-data', 'children'),
    [Input('filtered-accuracy-scatter', 'clickData'),
     Input('filtered-result-store', 'data')])
def update_model_click_data(clickData, result_data):
    #result_data = jsonpickle.decode(result_data)
    if clickData is not None:
        i = int(clickData['points'][0]['x'][1:])
        return result_data.convert_program_str_repr(result_data.model_list[i])

# ------   upload data section  ----------
def parse_contents_result(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'pkl' in filename:
            # initialize staff
            result = ResultProcessing("dataset/RuiJin_Processed.csv")
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


@cc.cached_callback([Output('ori-data-store', 'data')],
              [Input('upload-ori', 'contents')],
              [State('upload-ori', 'filename')])
def update_file_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    ori_data = parse_contents_ori(contents[0], filename[0])
    return ori_data

# --- end upload data section ---

@cc.callback([Output('main-visualization-content', 'children')],
              [Input('main-render-button', 'n_clicks'),
               Input('ori-data-store', 'data')])
def update_main_page(n_clicks, ori_data):
    names = ResultProcessing.read_dataset_names(ori_data)
    index_list = [i for i in range(len(names))]
    available_indicators = list(zip(index_list, names))
    if n_clicks == 0:
        raise PreventUpdate
    else:
        return render_main_visualization_layout(available_indicators)

@cc.callback([Output('updatemode-output-testing-acc', 'children'),
             ],
              [Input('testing-acc-filter-slider', 'value'),
               ])
def update_tesing_filter_value(value ):
    acc_text = str(value) + "% is used to filter models"
    return acc_text


@cc.callback([Output('updatemode-output-proglenfilter', 'children'),
              Output('ori_model_count', 'children')],
              [Input('prog-len-filter-slider', 'value'),
               Input('raw-result-store', 'data')])
def display_program_length_filter_value_and_ori_count(value, raw_result ):
    # program length filter --> effective features text display
    text = "Models with " + str(value) + " effective features are used"
    return text, str(len(raw_result.model_list))

@cc.callback([Output('prog-len-filter-slider', 'options')],
              [Input('filtered-result-store', 'data')])
def set_prog_len_radiobutton_and_update_filtered_data(result_data):
    result_data.calculate_featureList_and_calcvariableList()
    length_list = sorted(list(set([len(i) for i in result_data.feature_list])))
    return [{'label': str(i) , 'value': i} for i in length_list]

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
              Input('ori-data-store', 'data')])
def create_network(result_data, ori_data):
    names = ResultProcessing.read_dataset_names(ori_data)
    df = result_data.get_network_data(names)
    nodes = [
        {
            'data': {'id': node, 'label': node},
            'position': {'x': np.random.randint(0, 100), 'y': np.random.randint(0, 100)}
        }
        for node in df['source'].unique()
    ]
    edges = [
        {'data': {'source': df['source'][index], 'target': df['target'][index], 'weight': df['weight'][index]}}
        for index, row in df.iterrows()
    ]
    elements = nodes + edges
    return cyto.Cytoscape(
            id='cytoscape-layout-1',
            elements=elements,
            style={'width': '100%', 'height': '350px'},
            layout={
                'name': 'circle'
            },
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
                        'line-color': 'black'
                    }
                }
            ]
    )



cc.register(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True)
