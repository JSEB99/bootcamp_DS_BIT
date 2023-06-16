import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,plot, iplot
from dateutil.relativedelta import relativedelta # trabajar con fechas
from scipy.optimize import minimize              # 

import statsmodels.formula.api as smf            # statistics y econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


from itertools import product                    # algunas funciones utiles
from tqdm import tqdm_notebook

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima import auto_arima
import requests
from PIL import Image

import dash
import dash_bootstrap_components as dbc
from dash import dcc 
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output

df = pd.read_csv("solar_weather.csv")
df["Time"] = pd.to_datetime(df["Time"],format='%Y-%m-%d %H')
df["Time"].info()
df["Horario"] = df["Time"].dt.time # Extraemos horario 
ghi_diario = px.line(
    data_frame=df,
    x="Horario",
    y="GHI",
    color="Horario",
    title="Irradiancia durante el día",
    template="plotly_dark",
    labels= {'GHI':'Irradiancia Horizontal Global',
             'Horario':'Hora Diaria 24H'}
)
ghi_diario.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)),
                            xaxis=dict(nticks=23))
df["Fecha"] = df["Time"].dt.date
df_fecha = df[["Fecha","GHI"]].groupby("Fecha").sum()
df_fecha = df_fecha.reset_index()
df_fecha["Fecha"] = pd.to_datetime(df_fecha["Fecha"],format='%Y-%m-%d')
df_fecha["Year"] = df_fecha["Fecha"].dt.year
df_fecha_last = df_fecha[df_fecha["Year"]==2022]
fecha_ghi = px.line(
    data_frame=df_fecha_last,
    x="Fecha",
    y="GHI",
    labels={'Fecha':'Fecha del año 2022',
            'GHI':'Irradiancia Horizontal Global'},
    title="Irradiancia en el último año",
    template="plotly_dark"
)
fecha_ghi.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_fecha["Month"] = df_fecha["Fecha"].dt.month
df_fecha_month = df_fecha[["GHI","Month","Year"]].groupby(["Month","Year"]).mean()
df_fecha_month.reset_index(inplace=True)
df_fecha_month["Month"] = df_fecha_month["Month"].astype(str)
fecha_ghi_year = px.bar(
    data_frame=df_fecha_month,
    x="Year",
    y="GHI",
    title="Evolución Irradiancia Según el Mes",
    color="Month",
    template='plotly_dark',
    labels={"GHI":"Irradiancia Horizontal Global","Year":"Año"},
    barmode='group'
)
fecha_ghi_year.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_fecha_month_DE = df_fecha[["GHI","Month","Year"]].groupby(["Month","Year"]).std()
df_fecha_month_DE.reset_index(inplace=True)
df_fecha_month["STD"] = df_fecha_month_DE["GHI"]
df_fecha_month["Por_diff"] = round((1-(df_fecha_month["STD"]/df_fecha_month["GHI"]))*100)
error_bar = px.bar(
    data_frame=df_fecha_month,
    x="Year",
    y="Por_diff",
    title="Porcentaje de desviación estandar",
    color="Month",
    template='plotly_dark',
    labels={"Por_diff":"% desviación estandar","Year":"Año"},
    barmode='group'
)
error_bar.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_climate = df[["Time","temp","rain_1h","clouds_all","GHI","dayLength","hour","month","snow_1h"]]
df_climate["Fecha"] = df_climate["Time"].dt.date
df_climate["Horario"] = df_climate["Time"].dt.time
df_climate["Year"] = df_climate["Time"].dt.year
df_climate["dayLength"] = round(df_climate["dayLength"]/60)
dayduration = px.box(
    data_frame=df_climate,
    x="month",
    y="dayLength",
    template="plotly_dark",
    color="Year",
    boxmode="overlay",
    labels={'dayLength':'duración del día (Horas)',
            'month':'Mes','Year':'Año'},
    title="Duración del día durante el año"
)
dayduration.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_fecha["HSP"] = round(df_fecha["GHI"]/1000,2)
df_fecha["Year"] = df_fecha["Fecha"].dt.year
df_fecha["Year"] = df_fecha["Year"].astype(str)
df_fecha["Day"] = df_fecha["Fecha"].dt.day_of_year
HSP_ghi = px.bar(
    data_frame=df_fecha,
    x="Day",
    y="HSP",
    color="Year",
    labels={'Day':'día',
            'GHI':'Irradiancia Horizontal Global',
            'Year':"Año"
            },
    title="Evolución del HSP",
    template="plotly_dark",
    barmode='group'
)
HSP_ghi.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_climate["week"]=df_climate["Time"].dt.weekofyear
df_climate_clouds = df_climate[["clouds_all","week"]].groupby("week").mean()
nublaje = px.line(
    data_frame=df_climate_clouds,
    x=df_climate_clouds.index,
    y="clouds_all",
    labels={"x":"semana","clouds_all":"porcentaje de nubes"},
    title="Nublaje Promedio durante las Semanas del Año",
    template="plotly_dark"
)
nublaje.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_climate["Day"]=df_climate["Time"].dt.day_of_year
df_climate_rain_snow = df_climate[["rain_1h","snow_1h","Day","Year"]].groupby(["Year","Day"]).max()
df_climate_rain_snow.reset_index(inplace=True)
df_climate_rain_snow["Year"] = df_climate_rain_snow["Year"].astype(str)
lluvias = px.bar(
    data_frame=df_climate_rain_snow,
    x="Day",
    y="rain_1h",
    color="Year",
    template="plotly_dark",
    labels={"rain_1h":"lluvias (precipitación en mm)","Year":"Año","Day":"Día"},
    title="Precipitación máxima en los años",
    opacity=0.7,
    barmode="overlay"
)
lluvias.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
Nieves = px.bar(
    data_frame=df_climate_rain_snow,
    x="Day",
    y="snow_1h",
    color="Year",
    template="plotly_dark",
    labels={"snow_1h":"nevada (mm)","Year":"Año","Day":"Día"},
    title="Nevada máxima en los años",
    opacity=0.7,
    barmode="overlay"
)
Nieves.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_climate_temp = df_climate[["month","temp","Year"]].groupby(["Year","month"]).max()
df_climate_temp.reset_index(inplace=True)
df_climate_temp["GHI"]=df_fecha_month["GHI"]
temp = px.line(
    data_frame=df_climate_temp,
    x="month",
    y="temp",
    color="Year",
    labels={"Year":"Año","month":"Mes","temp":"temperatura °C"},
    title="Evolución de la temperatura máxima en el año",
    template="plotly_dark",
)
temp.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
tempGHI = px.scatter(
    data_frame=df_climate_temp,
    x="temp",
    y="GHI",
    color="temp",
    hover_data=["Year","month"],
    title="Irradiancia vs Temperatura",
    color_continuous_scale=px.colors.diverging.Portland,
    template="plotly_dark",
    labels={"month":"Mes","GHI":"Irradiancia Horizontal Global","temp":"Temperatura °C","Year":"Año"}
)
tempGHI.update_layout(title=dict(font=dict(family='JetBrains Mono',size=30)))
df_energy = df[["Energy delta[Wh]","temp","Time","GHI","hour","month"]]
df_energy["Horario"] = df_energy["Time"].dt.time
df_energy["Fecha"] = df_energy["Time"].dt.date
df_energy["Year"] = df_energy["Time"].dt.year
df_energy_ans = df_energy[["Energy delta[Wh]","Fecha","Year"]].groupby(["Fecha","Year"]).sum()
df_energy_ans_ghi = df_energy[["GHI","Fecha","Year"]].groupby(["Fecha","Year"]).sum()
df_energy_ans_temp = df_energy[["temp","Fecha","Year"]].groupby(["Fecha","Year"]).max()
df_energy_ans["GHI"] = df_energy_ans_ghi["GHI"]
df_energy_ans["temp"] = df_energy_ans_temp["temp"]
df_energy_ans.reset_index(inplace=True)
relation_byDate = px.scatter_3d(
    data_frame=df_energy_ans,
    x="Energy delta[Wh]",
    y="GHI",
    z="temp",
    color="temp",
    hover_data=["Fecha","Year"],
    title="Relación entre Consumo energético, Irradiancia y Temperatura °C Diaria",
    color_continuous_scale=px.colors.diverging.Portland,
    template="plotly_dark",
    labels={"Energy delta[Wh]":"Consumo energético diario [Wh]","GHI":"Irradiancia Horizontal Global","temp":"Temperatura °C","Year":"Año"}
)
relation_byDate.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))
relation_byHour_E = px.scatter(
    data_frame=df_energy,
    y="Energy delta[Wh]",
    x="Horario",
    color="Energy delta[Wh]",
    hover_data=["temp","Year","month"],
    title="Relación entre Consumo energético y la hora",
    color_continuous_scale=px.colors.diverging.Portland,
    template="plotly_dark",
    labels={"Energy delta[Wh]":"Consumo energético diario [Wh]","GHI":"Irradiancia Horizontal Global","temp":"Temperatura °C","Year":"Año",
            "month":"Mes"}
)
relation_byHour_E.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))
relation_byHour_T = px.scatter(
    data_frame=df_energy,
    y="temp",
    x="Horario",
    color="temp",
    hover_data=["temp","Year","month"],
    title="Relación entre temperatura y la hora",
    color_continuous_scale=px.colors.diverging.Portland,
    template="plotly_dark",
    labels={"Energy delta[Wh]":"Consumo energético diario [Wh]","GHI":"Irradiancia Horizontal Global","temp":"Temperatura °C","Year":"Año",
            "month":"Mes"}
)
relation_byHour_T.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))
relation_byHour_G = px.scatter(
    data_frame=df_energy,
    y="GHI",
    x="Horario",
    color="GHI",
    hover_data=["temp","Year","month"],
    title="Relación entre Irradiancia Horizontal Global y la hora",
    color_continuous_scale=px.colors.diverging.Portland,
    template="plotly_dark",
    labels={"Energy delta[Wh]":"Consumo energético diario [Wh]","GHI":"Irradiancia Horizontal Global","temp":"Temperatura °C","Year":"Año",
            "month":"Mes"}
)
relation_byHour_G.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))
df_air = df[["wind_speed","month","Time"]]
df_air["Year"] = df["Time"].dt.year
df_air_2 = df_air[["wind_speed","month","Year"]].groupby(["month","Year"]).max()
df_air_2.reset_index(inplace=True)
df_air_2["Year"] = df_air_2["Year"].astype(str)
air_speed = px.bar(
    data_frame=df_air_2,
    x="month",
    y="wind_speed",
    color="Year",
    title="Velocidad máxima de viento m/s",
    template="plotly_dark",
    labels={"Year":"Año","month":"Mes","wind_speed":"Velocidad de viento (m/s)"},
    barmode="group"
)
air_speed.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))

data = df
data['Fecha'] = data['Time'].dt.date
data_predict = data[['Fecha','GHI']]
data_predict["Fecha"] = pd.to_datetime(data_predict["Fecha"],format='%Y-%m-%d')
data_predict['DATE'] = data_predict['Fecha'].dt.date
data_predict = data_predict.groupby('DATE').sum()
data_predict.reset_index(inplace=True)
data_predict["DATE"] = pd.to_datetime(data_predict["DATE"],format='%Y-%m-%d')
data_predict["Fecha"] = data_predict['DATE'].dt.date
data_predict.drop(['DATE'],inplace=True,axis=1)

fecha_minima = data_predict['Fecha'].min()
fecha_maxima = data_predict['Fecha'].max()
rango_fechas = pd.date_range(start=fecha_minima, end=fecha_maxima)
fechas_faltantes = pd.DataFrame({'Fecha': rango_fechas})
fechas_faltantes = fechas_faltantes[~fechas_faltantes['Fecha'].isin(data_predict['Fecha'])]

data_predict.set_index('Fecha',inplace=True)
data_predict = data_predict.reindex(rango_fechas)  # Reindexar el DataFrame con todas las fechas
data_predict = data_predict.fillna(method='ffill')  # Llenar los valores faltantes con los valores cercanos
data_mensual = data_predict.reset_index()
data_mensual.rename(columns={'index':'Fecha'},inplace=True)
data_mensual["Fecha"] = pd.to_datetime(data_mensual["Fecha"])
data_mensual["Año"] = data_mensual["Fecha"].dt.year
data_mensual["Mes"] = data_mensual["Fecha"].dt.month
data_mensual = data_mensual.groupby(["Año", "Mes"]).mean()
data_mensual["Fechas"] = pd.to_datetime(data_mensual.index.get_level_values("Año").astype(str) + "-" + data_mensual.index.get_level_values("Mes").astype(str) + "-01")
data_mensual.set_index("Fechas", inplace=True)
# Graficamos los datos
fig1 = data_mensual.plot()
plt.title("Irradiancia promedio mensual")
plt.savefig("mensual.png")
decomposition = sm.tsa.seasonal_decompose(data_mensual, model = 'additive',period=12)
fig = decomposition.plot()
plt.savefig("decomp.png")

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_mensual['GHI'],lags=30,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_mensual['GHI'],lags=30,ax=ax2)
plt.savefig("Autocorrelation.png")

# Ajustar los parametros para dejar el valor mas bajo de AIC
d=0
D=1
s=12
#p, q, P, Q = result_table.parameters[0]
p, q, P, Q = 1,2,1,1

best_model=sm.tsa.statespace.SARIMAX(data_mensual.GHI, order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)

def mean_absolute_percentage_error(prediction,true_values):
    return np.mean(                                                      # Mean
                np.abs(                                                   # Absolute
                        prediction-true_values                            # Error
                    )/true_values
                )
def plotSARIMA(series, model, n_steps):
    """
        Plots del modelo vs valores predichos
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # Adicionar los valores del modelo
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # hacer un cambio en los pasos s + d, porque estos valores no fueron observados por el modelo
     # debido a la diferenciación
    data['arima_model'][:s+d] = np.NaN
    
    # pronosticando n_pasos adelante    
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    # calcular el error, nuevamente habiendo cambiado en los pasos s + d desde el principio
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(12, 6))
    plt.title("Mean Absolute Error: {0:.2f}%".format(error*100))# 
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True)

sns.set_style("whitegrid")
plotSARIMA(data_mensual, best_model, 100)
plt.savefig("Prediction.png")

url = "https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2017-01-01&end_date=2022-08-31&hourly=temperature_2m,direct_radiation,diffuse_radiation,direct_normal_irradiance"
response = requests.get(url=url)
data = response.json()

datax = data["hourly"]
dt = pd.DataFrame.from_dict(datax,orient="index").transpose()

dt["time"] = pd.to_datetime(dt["time"],format='%Y-%m-%d %H')
dt["Mes"] = dt["time"].dt.month
dt["Year"] = dt["time"].dt.year
dt["Day"] = dt["time"].dt.day_of_year
dt["Fecha"] = dt["time"].dt.date

dt_temp = dt[["Mes","temperature_2m","Year"]].groupby(["Mes","Year"]).max()
dt_temp.reset_index(inplace=True)
temp_2 = px.line(
    data_frame=dt_temp,
    x="Mes",
    y="temperature_2m",
    color="Year",
    title="Temperatura °C máxima según los meses",
    template="plotly_dark",
    labels={"temperature_2m":"Temperatura °C","Year":"Año"}
    )
temp_2.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))

dt_Irr = dt[["Day","direct_radiation","Year"]].groupby(["Day","Year"]).sum()
dt_Irr.reset_index(inplace=True)
Irr = px.line(
    data_frame=dt_Irr,
    x="Day",
    y="direct_radiation",
    color="Year",
    title="Irradiación directa W/m^2",
    template="plotly_dark",
    labels={"direct_radiation":"Radiación directa W/m^2","Year":"Año","Day":"día"}
    )
Irr.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))

dt_Irr_d = dt[["Day","diffuse_radiation","Year"]].groupby(["Day","Year"]).sum()
dt_Irr_d.reset_index(inplace=True)
dt_Irr["Radiation_diffuse"] = dt_Irr_d["diffuse_radiation"]
dt_Irr["GHI"] = dt_Irr["Radiation_diffuse"]+dt_Irr["direct_radiation"]
Irr_t = px.line(
    data_frame=dt_Irr,
    x="Day",
    y="GHI",
    color="Year",
    title="Irradiación Horizontal Global W/m^2",
    template="plotly_dark",
    labels={"GHI":"Irradiación Horizontal Global","Year":"Año","Day":"día"}
    )
Irr_t.update_layout(title=dict(font=dict(family='JetBrains Mono',size=20)))

mensual_image = Image.open("c:/Users/Usuario/Desktop/bootcamp_DS_BIT/Vis/env/mensual.png")
decomp_image = Image.open("c:/Users/Usuario/Desktop/bootcamp_DS_BIT/Vis/env/decomp.png")
autoc_image = Image.open("c:/Users/Usuario/Desktop/bootcamp_DS_BIT/Vis/env/Autocorrelation.png")
pred_image = Image.open("c:/Users/Usuario/Desktop/bootcamp_DS_BIT/Vis/env/Prediction.png")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "10px 10px 10px 10px",
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
                dbc.NavLink("Introducción",href="/",active="exact"),
                dbc.NavLink("Análisis Irradiancia", href="/page-1", active="exact"),
                dbc.NavLink("Análisis Condiciones Atmosféricas", href="/page-2", active="exact"),
                dbc.NavLink("Análisis de Consumos", href="/page-3", active="exact"),
                dbc.NavLink("Conexión API",href="/page-4",active="exact"),
                dbc.NavLink("Predicción de Irradiancia",href="/page-5",active="exact"),
                dbc.NavLink("Conclusiones",href="/page-6",active="exact"),
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
    if pathname =="/":
        return [
                html.H3("ANÁLISIS ENERGÉTICO SOLAR",style={"textAling":"center"}),
                html.P("""Mediante el siguiente análisis se busca dar a entender conceptos importantes 
                        de energía solar a traves del siguiente dataset que nos brinda cierta información para 
                        poder entender como se comportaría un sistema de fuente energética solar si se decidiera 
                        instalar."""),
                html.Br(),
                html.H5("PREGUNTAS/HIPÓTESIS",style={"textAling":"center"}),
                html.Br(),
                html.Li("""Se es posible la implementación de un sistema de energía solar fotovoltaica?"""),
                html.Li("""Que aspectos tocá tener en cuenta para su implementación?"""),
                html.Br(),
                html.H5("Contexto Análitico",style={"textAling":"center"}),
                html.Br(),
                html.P("""La empresa de MotoresGyM ha notado que sus facturas de luz le llegan demasiado elevadas
                y por tal motivo han tenido que recortar recursos de otras areas. Recientemente se contrato
                un ingeniero y se le pide que analice la viabilidad de usar energía solar en la empresa con el fin
                de reducir gastos a largo plazo y volver la industria mas sostenible."""),
                html.Br(),
                html.H5("Contexto Comercial",style={"textAling":"center"}),
                html.Br(),
                html.P("""La empresa necesita una forma de ahorrar su energía debido a que recientemente se han visto
                a pagar cantidades elevadas de dinero por la luz utilizada, esto en consecuencia de un aumento en 
                el precio en el sector industrial. Por tal motivo se busca una manera de usar las energías renovables para
                evitar este inconveniente."""),
                html.Br(),
                html.H5("Problema comercial",style={"textAling":"center"}),
                html.Br(),
                html.P("""De acuerdo a lo anterior el Gerente invita al ingeniero, y le plantea que tiene 
                       cierto dinero disponible para una instalación de paneles fotovoltaicos pero
                       no quiere arriesgarse a perder su dinero de no ser viable. Entonces le pregunta al ingeniero
                       lo siguiente: ¿Es viable la instalación de paneles solares?"""),
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Verificación de Irradiancia en el sitio',
                        style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                dcc.Graph(id='linegraph',
                         figure=ghi_diario),
                html.Br(),
                html.P("""Se presenta una baja irradiación global durante el día,
                dando a entender que es de un lugar donde el sol no esta muy presente 
                durante el año. Adémas la mayor irradiancia se da durante las horas de 
                la mañana y medio día"""),
                html.Br(),
                dcc.Graph(
                    id='linegraph2',
                    figure=fecha_ghi
                ),
                html.Br(),
                html.P("""Se observa que el sitio presenta cambios grandes en la irradiancia,
                lo que puede significar que el sitio tenga estaciones, se supone que en mitad
                de año presenta verano y en el inicio-final del año presenta inviernos."""),
                html.Br(),
                dcc.Graph(
                    id='bargraph',
                    figure=fecha_ghi_year
                ),
                html.Br(),
                html.P("""Se evidencia que en los años de muestra se tiene la misma tendencia
                de la irradiancia con picos máximos de aproximadamente 7kW/m^2 y un mínimo
                de 500W/m^2 en el peor mes de todos.
                """),
                html.Br(),
                dcc.Graph(
                    id='bargraph2',
                    figure=error_bar
                ),
                html.Br(),
                html.P("""Se presenta una variación bastante alta entre los datos, es decir,
                la irradiancia durante los meses no es constante y a su vez, esta varía durante
                el día. Por lo cual este recurso solo se puede aprovechar en ciertos momentos del 
                día. El promedio de desviación estandar es de 57% aproximadamente.""")
                ]
    elif pathname == "/page-2":
        return [
                html.H1('Verificación de condiciones atmosféricas',
                        style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                dcc.Graph(id='boxgrph',
                         figure=dayduration),
                html.Br(),
                html.P("""De acuerdo con lo anterior, se evidencia el tema de las estaciones
                donde a mitad de año se presentan días mas largos y en los eneros-diciembres
                se presentan dias mas cortos. Con un máximo de 17 horas de día y un mínimo de
                8 horas."""),
                html.Br(),
                dcc.Graph(
                    id='bargraph3',
                    figure=HSP_ghi
                ),
                html.Br(),
                html.P("""A pesar de tener altas horas de día en ciertas epocas, en temás
                energéticos solares fotovoltaicos existe un dato importe que son las HSP
                (Horas sol pico) este dato se realiza diviendo la irradiancia día sobre
                1kW/m^2,que es medida estandar de operatividad de un panel común en el 
                mercado. En este caso un máximo de 8.68 Horas y un mínimo de 0.14 hora de 
                efectividad para proyectos solares fotovoltaicos."""),
                html.Br(),
                dcc.Graph(
                    id='linegraph3',
                    figure=nublaje
                ),
                html.Br(),
                html.P("""Según el dataset nos proporciona una medida de porcentaje de
                nublaje del sitio, sin embargo, se evidencia que en los meses de enero y
                diciembre existe mas porcentaje de nubosidad."""),
                html.Br(),
                dcc.Graph(
                    id='bargraph4',
                    figure=lluvias
                ),
                html.Br(),
                html.P("""A pesar de que en mitad de año es donde se presenta menos nubosidad,
                es donde más lluvias se presentan. (Unidades en mm)"""),
                html.Br(),
                dcc.Graph(
                    id='bargraph5',
                    figure=Nieves
                ),
                html.Br(),
                html.P("""En cuanto a nevadas si concuerda con las suposiciones de los inviernos,
                donde se presentan nevadas en enero-diciembre."""),
                html.Br(),
                dcc.Graph(
                    id='linegraph4',
                    figure=temp
                ),
                html.Br(),
                dcc.Graph(
                    id='scattergraph',
                    figure=tempGHI
                ),
                html.Br(),
                html.P("""Se presenta la temperatura en el año, donde se evidencian los cambios 
                de estaciones con un máimo de 35°C y un mínimo de 5°C aproximadamente. Ademas, se 
                relaciona la irradiancia con la temperatura, donde se evidencia que claramente no 
                es un factor importante en la irradiancia del sitio.""")
                ]
    elif pathname == "/page-3":
        return [
                html.H1('Verificación entre consumos del sitio',
                        style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                dcc.Graph(
                    id='scatter3d',
                    figure=relation_byDate
                ),
                html.Br(),
                html.P("""Se identifica que a pesar de ser independientes de la temperatura, la 
                Irradiancia es directamente proporcional al consumo energético diario en la gran 
                mayoria de los casos"""),
                html.Br(),
                dcc.Graph(
                    id='scatter1',
                    figure=relation_byHour_E
                ),
                html.Br(),
                html.P("""Existe una relación entre Consumo energético y el horario del medio día, 
                que coincide con las horas de mayor temperatura y de mayor irradiancia"""),
                html.Br(),
                dcc.Graph(
                    id='scatter2',
                    figure=relation_byHour_T
                ),
                html.Br(),
                html.P("""Se verifica la nota anterior, sin embargo esto varía con ciertos años ya 
                que en 2019 fue donde mas altas temperaturas se presentaron"""),
                html.Br(),
                dcc.Graph(
                    id='scatter3',
                    figure=relation_byHour_G
                ),
                html.Br(),
                html.P("""Finalmente se evidencia mayor irradiancia en las horas del medio día"""),
                html.Br(),
                dcc.Graph(
                    id='bargraph5',
                    figure=air_speed
                ),
                html.Br(),
                html.P("""Como alternativa se presentan las relaciones de velocidad de viento en los
                años medidos para presentar de manera alterna una propuesta de aerogeneradores.""")
                ]
    elif pathname == "/page-4":
        return [
                html.H1('Conexión API',
                        style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                html.P("""Para este apartado se selecciona una API de información de irradiancia solar y temperatura"""),
                html.P("https://open-meteo.com/en/docs/historical-weather-api#latitude=52.52&longitude=13.41&start_date=2017-01-01&end_date=2022-08-31&hourly=temperature_2m,direct_radiation,diffuse_radiation,direct_normal_irradiance"),
                html.Br(),
                dcc.Graph(
                    id='linegraph5',
                    figure=temp_2
                ),
                html.Br(),
                dcc.Graph(
                    id='linegraph6',
                    figure=Irr
                ),
                html.Br(),
                dcc.Graph(
                    id='linegraph7',
                    figure=Irr_t
                ),
                html.Br(),
                html.P("""Como se desarrollo la anterior información se puede aplicar 
                con los datos establecidos en este dataset por tal motivo se realizaran 
                unos gráficos de temperatura y radiación directa a lo largo del tiempo 
                para evidenciar su comportmiento en este lugar."""),
                html.Br(),
                html.P("""Se obtienen datos similares que en el anterior dataframe, la 
                principal diferencia radica en que no obtiene datos tan altos de irradiancia, 
                sin embargo en este dato falta sumar la irradiancia difusa.
                """),
                html.Br(),
                html.Li("""Se presenta una mejoría en la radiación respecto al anterior lugar por lo cual 
                es mas apto para uso de energía solar fotovoltaica y térmica"""),
                html.Li("""Se evidencian condiciones similares en cuanto a que tiene estaciones y cambios 
                de temperatura dentro de los mismos rangos""")
                ]
    elif pathname == "/page-5":
        return [
                html.H1('Predicción de Irradiancia',
                        style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                html.Br(),
                html.P("""Se procede a verificar los datos de irradiancia, una de las variables
                        mas importantes en un diseño solar fotovoltaico """),
                html.Div(
                    children=[
                        html.Img(src=mensual_image,style={"width": "800px", "height": "500px","margin":"auto"}),
                            ]
                        ),
                html.Br(),
                html.P("""Se identifica que los datos son una serie de tiempo, se plantea
                        realizar una predicción por medio del modelo SARIMAX\nLos modelos
                        SARIMAX (auto regresivos integrados de medias móviles estacional
                        con variables explicativas) son una estructura matemática que contiene todas las bondades de
                        un modelo SARIMA adicionalmente, pueden capturar información sobre variables exógenas que 
                        permiten a entender y pronosticar la variable de interés. Para una única variable de interés y
                        , y una única variable independiente x
                        ."""),
                html.Br(),
                html.P("""Se verifican los datos con una descomposición"""),
                html.Br(),
                html.Img(src=decomp_image,style={"width": "800px", "height": "500px"}),
                html.Br(),
                html.P("""Se verifica la autocorrelación y autocorrelación
                parcial para una serie de lags"""),
                html.Br(),
                html.Img(src=autoc_image,style={"width": "1050px", "height": "700px"}),
                html.Br(),
                html.P("""Se verifica que la gráfica es estacionaria, tiene estacionalidad mensual, no 
                tiene picos que generen impacto en la autocorrelación, pero si tiene en la autocorrelación
                parcial y no se le hicieron diferencias. De acuerdo a lo anterior el modelo SARIMAX 
                necesita de unos parámetros, estos parámetros son: (q -> 1,d -> 0,p -> 2)
                 (Q->1,D-> 1,P-> 1,s-> 12), con ellos se realiza la predicción del modelo """),
                html.Br(),
                html.Img(src=pred_image,style={"width": "1050px", "height": "700px"}),
                html.Br(),
                html.P("""Obteniendo unos resultados bastante buenos y que indican que la Irradiancia en 
                la ubicación se comporta como una Onda sinusoidal, dando idea de que eppocas son aptas para
                la instalación de un módulo fotovoltaico."""),
            ]
    elif pathname == "/page-6":
        return [
                html.H1("Análisis y Conclusiones principales",style={'textAlign':'center',
                               "font-family":"Coolvetica",
                                "font-size":40,
                                "color":"#161616",}),
                html.H5("Autor: J. Sebastian Mora T. / Estudiante de Maestría en Energías Renovables",style={"textAling":"center"}),
                html.Br(),
                html.H5("Respuestas a preguntas",style={"textAling":"center"}),
                html.Br(),
                html.H5("Contexto Análitico",style={"textAling":"center"}),
                html.P("""La implementación es posible en fechas de verano donde se tienen optimas condiciones
                 para la instalación, sin embargo hay que tener presente que las temperaturas pueden llegar a 
                afectar el sistema. Por tal motivo es indispensable tener sistemas de refrigeración en dichas 
                epocas donde la temperatura alcaza los 35 grados celcios. Además, se deben diseñar sistemas de 
                acuerdo a un consumo del medio día de las personas que es donde mas se presenta su uso."""),
                html.P("Consideraciones para diseño:"),
                html.Li("""Horas solares pico maximas = 8"""),
                html.Li("""Temperatura máxima = 35 °C"""),
                html.Li("""Irradiancia máxima = 8 kW/M2 (día)"""),
                html.Li("""Se deben tener inversores de buena cálidad ya que en el medio día es donde se presentan 
                cargas AC como licuadoras, hornos, microondas (hora de almuerzo) y considerar que por esas horas 
                aun funcionan las industrias por lo cual se deben tener inversores de buena calidad"""),
                html.H5("Contexto Comercial",style={"textAling":"center"}),
                html.Br(),
                html.P("""Se tienen datos establecidos para implementación de sistemas fotovoltaicos en el municipio, 
                a pesar de tener fechas buenas como mitad de año y cercanos, se pueden llegar a implementar sistemas 
                térmicos por las altas temperaturas que se llegan a obtener. Para fechas invernales no se presentan 
                buenas condiciones para su implementación debido a la escacez de energía, sin embargo se obtienen buenas 
                cantidad de velocidad de viento para implementación de aerogeneradores como alternativa de diseño."""),
                html.H5("Respuesta problema comercial",style={"textAling":"center"}),
                html.Br(),
                html.P("""Se presentan considerciones optimas para la instalación y operación desde Marzo a Octubre,
                en las fechas de los extremos del año es mejor buscar alternativas de energía eolica o reducir la producción.
                Sin embargo, para la producción anual se tiene un amplio margen de uso de las energías solares fotovoltaicas,
                por lo cual si es viable la instalación. """),
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
    app.run_server(debug=True)

