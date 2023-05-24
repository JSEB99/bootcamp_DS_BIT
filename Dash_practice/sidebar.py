import dash
import dash_bootstrap_components as dbc
from dash import dcc 
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Bootstrap/Side-Bar/iranian_students.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "40px 10px 10px 10px",
    "background-color": "#161616",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "1px",
    "padding": "1px 1px 10px 10px",
    "background-color":"#F1AD25" 
}

sidebar = html.Div(
    [
        html.H2("ANÁLISIS SISTEMA RENOVABLE", className="font-weight bolder",
                style={
                    "font-family":"Coolvetica",
                    "font-size":40,
                    "color":"#F1AD25",
                    }),
        html.Br(),
        html.H3("Autor: Juan Sebastian Mora Tibamoso",className="blockquote",
                style={
                    "color":"#FAFAFA"
                }),
        html.H5("Datos: https://www.kaggle.com/datasets/samanemami/renewable-energy-and-weather-conditions",className="blockquote-footer"),
        html.Br(),
        html.P(
            """Análisis de dataset de consumos energéticos, se revisan condiciones de irradiancia
            ,condiciones atmosféricas y condiciones de consumo de la localidad.
            """, className="lead",style={"color":"#FAFAFA"}
        ),
        html.Br(),
        dbc.Nav(
            [
                dbc.NavLink("Análisis Irradiancia", href="/", active="exact"),
                dbc.NavLink("Análisis Condiciones Atmosféricas", href="/page-1", active="exact"),
                dbc.NavLink("Análisis de Consumos", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
            justified=False,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Kindergarten in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Kindergarten', 'Boys Kindergarten'])),
                html.Br(),
                html.P("""Lorem Ipsum es simplemente el texto de relleno
                de las imprentas y archivos de texto. Lorem Ipsum ha sido
                el texto de relleno estándar de las industrias desde el 
                año 1500, cuando un impresor (N. del T. persona que se 
                dedica a la imprenta) desconocido usó una galería de textos
                y los mezcló de tal manera que logró hacer un libro de textos
                especimen. No sólo sobrevivió 500 años, sino que tambien
                ingresó como texto de relleno en documentos electrónicos, 
                quedando esencialmente igual al original. Fue popularizado en 
                los 60s con la creación de las hojas "Letraset", las cuales 
                contenian pasajes de Lorem Ipsum, y más recientemente con 
                software de autoedición, como por ejemplo Aldus PageMaker, 
                el cual incluye versiones de Lorem Ipsum."""),
                html.Br(),
                html.P("""Lorem Ipsum es simplemente el texto de relleno
                de las imprentas y archivos de texto. Lorem Ipsum ha sido
                el texto de relleno estándar de las industrias desde el 
                año 1500, cuando un impresor (N. del T. persona que se 
                dedica a la imprenta) desconocido usó una galería de textos
                y los mezcló de tal manera que logró hacer un libro de textos
                especimen. No sólo sobrevivió 500 años, sino que tambien
                ingresó como texto de relleno en documentos electrónicos, 
                quedando esencialmente igual al original. Fue popularizado en 
                los 60s con la creación de las hojas "Letraset", las cuales 
                contenian pasajes de Lorem Ipsum, y más recientemente con 
                software de autoedición, como por ejemplo Aldus PageMaker, 
                el cual incluye versiones de Lorem Ipsum.""")
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Grad School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Grade School', 'Boys Grade School']))
                ]
    elif pathname == "/page-2":
        return [
                html.H1('High School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls High School', 'Boys High School']))
                ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__=='__main__':
    app.run_server(debug=True, port=3000)
