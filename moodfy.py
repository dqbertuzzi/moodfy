# Standard library imports
import os
import base64
import requests
from requests import post, get
from concurrent.futures import ThreadPoolExecutor
import json

# Data manipulation and analysis imports
import numpy as np
from numpy.linalg import norm
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Visualization and plotting imports
import plotly.graph_objs as go
import plotly.express as px

# Dash and related imports
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc

# Other specific library imports
from dotenv import load_dotenv

load_dotenv()

# Spotify API related functions

def get_token(client_id, client_secret):
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization":"Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_tracks(playlist_url, headers):
    playlist_id = playlist_url.split('/')[-1].split("?")[0]
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks?offset=0'

    response = get(url=url, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return "Erro! O link da playlist não é válido."
    
    data = response.json()
    
    tracks_data = data['items']
    tracks = pd.DataFrame({
        'artist': [item['track']['artists'][0]['name'] for item in tracks_data],
        'album_image': [item['track']['album']['images'][0]['url'] for item in tracks_data],
        'track_title': [item['track']['name'] for item in tracks_data],
        'track_href': [item['track']['external_urls']['spotify'].split('/')[-1] for item in tracks_data]
    })
    return tracks

def get_audio_features(href, headers):
    url = f'https://api.spotify.com/v1/audio-features/{href}'

    response = get(url=url, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return "429"
    audio_features = response.json()
    return audio_features

def get_audio_features_parallel(track_hrefs, headers):
    with ThreadPoolExecutor(max_workers=10) as executor:
        audio_features = list(executor.map(lambda h: get_audio_features(h, headers), track_hrefs))
    return audio_features

def get_final_dataset(playlist_id, headers):
    tracks = get_tracks(playlist_id, headers)
    if isinstance(tracks, str):
        return tracks
    
    track_hrefs = tracks['track_href'].tolist()
    audio_features = get_audio_features_parallel(track_hrefs, headers)
    if audio_features[0]=='429':
        return audio_features[0]
    
    final_dataset = pd.concat([tracks, pd.DataFrame(audio_features)], axis=1)
    return final_dataset


# Data manipulation and scaling functions
def scale_columns(final_dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    columns_to_scale = ['valence', 'energy']
    
    scaled_data = np.round(scaler.fit_transform(final_dataset[columns_to_scale]),2)
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in columns_to_scale])
    
    final_dataset = pd.concat([final_dataset, scaled_df], axis=1)
    
    return final_dataset

def quadrant(x, y):
    if x >= 0 and y >= 0:
        return [[x,y],"Q1", "#FFCA3A"]
    elif x < 0 and y >= 0:
        return [[x,y],"Q2", "#FF595E"]
    elif x < 0 and y < 0:
        return [[x,y],"Q3", "#1982C4"]
    elif x >= 0 and y < 0:
        return [[x,y],"Q4", "#8AC926"]
    else:
        return [[x,y], "Neutro", "#160C28"]

def calculate_metrics(final_dataset):
    columns_scaled = ['valence_scaled', 'energy_scaled']
    
    median = np.round(np.median(final_dataset[columns_scaled], axis=0),2)
    playlist_mood = quadrant(median[0], median[1])
    
    # Calculate score
    A = np.array(final_dataset[columns_scaled])
    final_dataset['cos_similarity'] = 1 - np.round(np.dot(A,median)/(norm(A, axis=1)*norm(median)), 2)
    final_dataset['distance'] = np.round(norm(A - median, axis=1), 2)
    final_dataset['score'] = (final_dataset['distance']) * (final_dataset['cos_similarity']) * (-1)
                                                     
    # Calculate outliers
    outlier_fraction = 0.05
    threshold = stats.scoreatpercentile(final_dataset['score'], 100 * outlier_fraction)
    final_dataset['outlier'] = np.where(final_dataset['score'] < threshold, 1, 0)
    
    return final_dataset, playlist_mood


# Visualization functions
def create_mood_scatter_plot(playlist_mood, outlier_data):
    # Create scatter plot
    fig = px.scatter(x=[playlist_mood[0][0]], y=[playlist_mood[0][1]], labels={"x": "Valência", "y": "Energia"},
                     template="simple_white", width=800, height=800)
    
    # Update layout
    fig.update_layout(
        yaxis_range=[-1.1, 1.1],
        xaxis_range=[-1.1, 1.1],
        yaxis=dict(visible=True, showticklabels=False),
        xaxis=dict(visible=True, showticklabels=False)
    )

    # Add arrows
    x_end = [0.9, 0]
    y_end = [0, 0.9]
    x_start = [-0.9, 0]
    y_start = [0, -0.9]
    list_of_all_arrows = []

    for x0, y0, x1, y1 in zip(x_end, y_end, x_start, y_start):
        arrow = go.layout.Annotation(dict(
            x=x0,
            y=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            ax=x1,
            ay=y1,
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1.5,
            arrowcolor="#636363"
        ))

        list_of_all_arrows.append(arrow)

    fig.update_layout(
        annotations=list_of_all_arrows,
        font_family="Mukta Vaani",
        xaxis_title="VALÊNCIA (→)",
        yaxis_title="ENERGIA (→)",
        font=dict(size=24)
    )

    # Add text annotations
    x = [0, 0, -0.9, 0.9]
    y = [0.9, -0.9, 0, 0]
    xshift = [35, 35, 35, -35]
    yshift = [-25, 25, 25, 25]
    text = ["<i>Alta</i>", "<i>Baixa</i>", "<i>Negativa</i>", "<i>Positiva</i>"]

    for x_, y_, t_, xshift_, yshift_ in zip(x, y, text, xshift, yshift):
        fig.add_annotation(x=x_, y=y_,
                           text=t_,
                           showarrow=False,
                           yshift=yshift_,
                           xshift=xshift_,
                           font={"size": 14})

    x = [0.5, -0.5, -0.5, 0.5]
    y = [0.5, -0.5, 0.5, -0.5]
    text = ["<b>Felicidade</b>", "<b>Tristeza</b>", "<b>Tensão</b>", "<b>Tranquilidade</b>"]

    for x_, y_, t_ in zip(x, y, text):
        fig.add_annotation(x=x_, y=y_,
                           text=t_,
                           showarrow=False,
                           font={"size": 24},
                           align="center",
                           opacity=0.25)

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    fig.update_traces(
        marker=dict(size=50, color=playlist_mood[2], line=dict(width=1, color='#160C28')),
        selector=dict(mode='markers'),
        hovertemplate="<br>".join([
            "Playlist",
            "Valência: %{x}",
            "Energia: %{y}"
        ])
    )

    fig2 = px.scatter(data_frame = outlier_data, x='valence_scaled', y='energy_scaled', labels={"x": "Valência", "y": "Energia"}, custom_data=['artist', 'track_title'])
    fig2.update_traces(
        hovertemplate="<br>".join([
            "Artista: %{customdata[0]}",
            "Faixa: %{customdata[1]}",
            "Valência: %{x}",
            "Energia: %{y}"
        ])
    )
    fig2.update_traces(
        marker=dict(size=10, color='black', line=dict(width=0, color='#160C28')),
        selector=dict(mode='markers')
    )
    
    fig.add_traces(
        list(fig2.select_traces())
    )

    return fig

app = Dash(__name__,
          meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}], external_stylesheets=[dbc.themes.MINTY])
server = app.server
app.config.suppress_callback_exceptions = True

def display_artistInfo(info):
    elements = [html.H2("Estas são as faixas que não combinam com a emoção da playlist:",
            style={'color':'#191414',"font-size":"2.8vh"})]
    for e in range(len(info['artists'])):
        elements.append(html.Br())
        elements.append(html.Img(src=info['img_srcs'][e], style={'height':'35px', 'width':'35px', 'vertical-align': 'middle', 'display':'inline-block'}))
        elements.append(html.Div(html.P(info['artists'][e] + " - " + info['track_titles'][e]),style={'display': 'inline-block', 'padding-left': '.5vw'}))
        
    return elements

tab_style = {
    'font-size': '2.7vh'
}

tab_selected_style = {
    'backgroundColor': '#191414',
    'color': 'white',
}

app.layout = html.Div([
    dcc.Store(id='dataset', storage_type='memory'),
    dcc.Store(id='playlistMood', storage_type='memory'),
    html.Div(
    html.H1("Spotify Mood Check",
            style={'color':'#1db954', "font-size":"4vh"}), style={'display': 'inline-block'}
            ),
    html.Div(
    html.Hr(style={'border':'none', 'height':'.3vh', 'background-color':'#160C28', 'display': 'block'})),
    html.H2("Sintonize sua emoção: descubra o sentimento por trás da sua playlist!",
            style={'color':'#191414',"font-size":"2.8vh", "margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw','line-break': 'strict'}),
    dbc.Tabs([
        dbc.Tab([
            html.Br(),
            dcc.Input(
                id="input_playlistUrl",
                type='text',
                placeholder="Link da playlist", style={'width':'60vw','display': 'inline-block', 'vertical-align': 'middle'}
                ),
            html.Div([
                dbc.Button("Enviar", color="dark", className="me-1", id='submit-button')],
                style={'margin-left':'1vw',
                       'display': 'inline-block',
                       'vertical-align': 'middle'}
                       ),
            dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Servidor sobrecarregado!")),
                       dbc.ModalBody(["Volte em alguns minutos e tente de novo."])],
            id='modal1', is_open=False),
            dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Link inválido!")),
                       dbc.ModalBody([html.Ul([html.Li("O link deve estar no formato https://open.spotify.com/playlist/XXXXXXXXXXXXXXXXXXXXXX?si=XXXXXXXXXXXXXXXX"),
                                               html.Li("A playlist deve estar configurada como pública.")])])],
                       id='modal2', is_open=False, size="lg",),
            html.Div([
                dbc.Spinner(html.Div(id="loading-output-1",
                                     style={'display': 'inline-block', 'padding-left': '50px'}))],
                                     style={'display': 'inline-block','vertical-align': 'middle'}),
            html.Br(),
            html.Div([
                html.Br(),
                html.P("O gráfico exibe onde sua playlist se situa emocionalmente (círculo maior colorido) e as faixas musicais que não combinam (círculos menores pretos):",
                        id='mood-h2', style={'display':'none', 'color':'#191414',"font-size":"3vh", 'margin-left':'1.5vw','margin-right':'1.5vw','line-break': 'strict'}),
                dcc.Graph(id='graph-figure', style={'display': 'none'})],
                id="mood-div"),
            html.Div(id='output-album-image', style={
                                                     'display': 'inline-block',
                                                     'vertical-align': 'middle', "margin-bottom":"2vh", 'margin-left':'1.5vw'}
                                                     )], label='Análise', tab_style=tab_style),
        dbc.Tab([
            html.Div([
            html.Br(),
            html.H2("Sobre o Modelo Circumplexo de Emoção", style={'color':'#191414',"font-size":"3vh"}),
            html.P("Na década de 1980, Russell (1980) apresentou uma contribuição teórica para a compreensão do afeto, por meio da caracterização de duas dimensões: valência e ativação. As combinações dessas duas dimensões, em diferentes graus, teriam como resultado as experiências afetivas. O modelo teórico resultante, denominado de circumplexo de Russell (1980), teve continuidade em estudos posteriores (Carroll, Yik, Russell, & Barrett, 1999; Russell & Barrett, 1999; Russell, 2003; Yik, Russell, & Steiger, 2011). Estudos como os de Carroll et al. (1999) e Yik et al. (2011) trouxeram avanços com relação a como modelar as variáveis de afeto em um circumplexo e como sentimentos são entendidos por esse modelo; pesquisas como a de Russell e Barrett (1999) visavam indicar novas perspectivas sobre como medir afeto e discutir esse construto teoricamente. Ao longo das décadas houve mudanças importantes na teoria do afeto, especialmente ao se reconhecer que os estudos sobre humor pertenciam ao campo conceitual do afeto.", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P("O afeto, para Russell (1980), é compreendido por meio do circumplexo (Figura 1). Suas dimensões são bipolares e ortogonais, sendo nomeadas de valência (prazer ou desprazer) e ativação percebida (alta ou baixa). O circumplexo é uma estrutura ordenada em que todos os testes apresentam um mesmo nível de complexidade e diferem em termos do tipo de habilidade que eles medem. Quando um construto pode ser representado por um circumplexo, sua matriz de correlações apresenta um padrão de correlações fortes perto da diagonal e, conforme as correlações se afastam da diagonal, elas ficam mais fracas, até que voltam a ficar fortes. Esse padrão de correlações repete-se em toda a matriz, e, por isso, pontos próximos no circumplexo são correlacionados fortemente (Guttman, 1954).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P([html.Img(src='http://pepsic.bvsalud.org/img/revistas/avp/v16n2/n2a05f1.jpg', style={'height':'300px', 'width':'300px'})], style={"margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw', 'text-align':'center'}),
            html.P("A dimensão valência está relacionada à codificação do ambiente como prazeroso ou desprazeroso. Para o estímulo em um determinado momento, o sujeito pode atribuir um significado: bom ou ruim; útil ou prejudicial; recompensador ou ameaçador (Barrett, 2006). A ativação, por sua vez, é a dimensão da experiência que corresponde à mobilização ou energia dispensada; ou seja, é representada por um continuum, desde a baixa ativação, representada por sono, até a ativação alta, representada pela excitação (Russell & Barrett, 1999).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P("Estados afetivos que são próximos no circumplexo representam uma combinação similar de valência e ativação percebida; já estados afetivos posicionados diametricamente longe um do outro diferem em termos de valência e ativação (Russell, 1980). Assim, as quatro variáveis alocadas diagonalmente não são dimensões, mas ajudam a definir os quadrantes no espaço do circumplexo (Russell, 1980).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P(html.A("Crispim, Ana Carla, Cruz, Roberto Moraes, Oliveira, Cassandra Melo, & Archer, Aline Battisti. (2017). O afeto sob a perspectiva do circumplexo: evidências de validade de construto. Avaliação Psicológica, 16(2), 145-152. https://dx.doi.org/10.15689/AP.2017.1602.04", href='http://dx.doi.org/10.15689/AP.2017.1602.04', style={"font-size":"2vh",'text-align':'justify'}),
                   style={'padding-left':'15vw', 'text-align':'justify'}), 
        ], style={"margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw', 'line-break': 'strict'})
        ], label="Saiba mais", tab_style=tab_style)])
    ]
    ,
    style={'margin-left': '2.5vw',
           'margin-top': '2.5vw',
           'margin-bottom': '2.5vw',
           'margin-right':'2.5vw'}
)

@callback(
        [Output("loading-output-1", "children"),
        Output("modal1", "is_open"),
        Output("modal2", "is_open"),
        Output("dataset", 'data'),
        Output("playlistMood", 'data')],
        Input('submit-button', 'n_clicks'),
        State('input_playlistUrl', 'value'),
        prevent_initial_call=True
        )
def update_output(clicks, input_value):
    if clicks is not None:
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        token = get_token(client_id, client_secret)
        headers = get_auth_header(token)
        
        url = f'{input_value}'
        
        final_dataset = get_final_dataset(url, headers)

        if final_dataset=='429':
            return None, True, False, no_update, no_update

        elif  isinstance(final_dataset, str):
            return None, False, True, no_update, no_update
        
        else:
            final_dataset = scale_columns(final_dataset)
            final_dataset, playlist_mood = calculate_metrics(final_dataset)

            return None, False, False, final_dataset.set_index('track_href').to_dict('records'), playlist_mood
        
@callback(
    [Output('graph-figure', 'figure'),
     Output('graph-figure', 'style'),
     Output('output-album-image', 'children'),
     Output('mood-h2', 'style')],
     Input('dataset', 'data'),
     Input('playlistMood', 'data'),
     prevent_initial_call=True
     )
def display_results(dataset, playlistMood):
    dff = pd.DataFrame(dataset)

    outlier_data = dff[dff['outlier'] == 1]
    outlier_artists = {'artists': list(outlier_data['artist']),
               'track_titles': list(outlier_data['track_title']),
               'img_srcs': list(outlier_data['album_image'])}

    return create_mood_scatter_plot(playlistMood, outlier_data), {'display': 'block'}, display_artistInfo(outlier_artists), {'display': 'block'}
    
if __name__ == "__main__":
    app.run(debug=True)
