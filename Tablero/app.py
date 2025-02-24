import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import pickle
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo entrenado  
with open("modelo_polynomial.pkl", "rb") as f:
    coeficientes, intercepto, poly, X_scaler3, y_scaler3, nombres_features = pickle.load(f)

# Cargar los datos de apartamentos
file_path = "datos_apartamentos_rent.csv"
df = pd.read_csv(file_path, encoding="latin1", sep=";")

# Eliminar valores NaN en las columnas 'state' y 'cityname'
df = df.dropna(subset=['state', 'cityname'])

# Aplicar LabelEncoder a las columnas 'state' y 'cityname'
state_encoder = LabelEncoder()
df['state_encoded'] = state_encoder.fit_transform(df['state'])
city_encoder = LabelEncoder()
df['city_encoded'] = city_encoder.fit_transform(df['cityname'])

# Obtener los estados y ciudades únicos con sus valores asignados
state_options = [{'label': state, 'value': code} for state, code in zip(state_encoder.classes_, state_encoder.transform(state_encoder.classes_))]
city_options = [{'label': city, 'value': code} for city, code in zip(city_encoder.classes_, city_encoder.transform(city_encoder.classes_))]

# Opciones de fuentes de publicación
source_sites = {
    "source_ListedBuy": "ListedBuy",
    "source_RentDigs.com": "RentDigs.com",
    "source_RentLingo": "RentLingo",
    "source_Listanza": "Listanza"
}
source_options = [{'label': name, 'value': key} for key, name in source_sites.items()]

# Crear un mapa de selección de estados sin la barra de color
fig = px.choropleth(
    df,
    locations="state",
    locationmode="USA-states",
    scope="usa",
    title="Seleccione un estado en el mapa"
)
fig.update_layout(coloraxis_showscale=False)  # Eliminar la barra de color

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Diseño de la aplicación
app.layout = dbc.Container([
    html.H1("Predicción precio de un apartamento"),
    dcc.Graph(id="state-map", figure=fig),
    dcc.Store(id="selected-state"),
    html.Label("Ingrese la latitud:"),
    dcc.Input(id='input-latitude', type='number', value=0, step=0.01),
    html.Label("Ingrese la longitud:"),
    dcc.Input(id='input-longitude', type='number', value=0, step=0.01),
    html.Label("Ingrese la cantidad de baños:"),
    dcc.Input(id='input-bathrooms', type='number', value=1, step=1),
    html.Label("Ingrese la cantidad de habitaciones:"),
    dcc.Input(id='input-bedrooms', type='number', value=1, step=1),
    html.Label("Ingrese los pies cuadrados:"),
    dcc.Input(id='input-square-feet', type='number', value=200, step=10),
    html.Label("Seleccione el estado:"),
    dcc.Dropdown(id='input-state', options=state_options, value=state_options[0]['value']),
    html.Label("Seleccione la ciudad:"),
    dcc.Dropdown(id='input-city', options=city_options, value=city_options[0]['value']),
    html.Label("¿Se permiten mascotas?"),
    dcc.RadioItems(id='input-pets', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
    html.Label("Seleccione la fuente de publicación:"),
    dcc.Dropdown(id='input-source', options=source_options, value=list(source_sites.keys())[0]),
    html.Label("Seleccione la categoría del inmueble:"),
    dcc.RadioItems(id='input-category', options=[
        {'label': 'Vivienda en renta', 'value': 'home'},
        {'label': 'Alquiler a corto plazo', 'value': 'short_term'}
    ], value='home'),
    html.Label("¿Tiene fotos?"),
    dcc.RadioItems(id='input-photos', options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}], value=1),
    html.Button('Predecir', id='predict-button', n_clicks=0),
    html.H3(id='prediction-output', style={'marginTop': '20px'})
])

# Callback para actualizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('input-latitude', 'value'),
    Input('input-longitude', 'value'),
    Input('input-bathrooms', 'value'),
    Input('input-bedrooms', 'value'),
    Input('input-square-feet', 'value'),
    Input('input-state', 'value'),
    Input('input-city', 'value'),
    Input('input-pets', 'value'),
    Input('input-source', 'value'),
    Input('input-category', 'value'),
    Input('input-photos', 'value')
)
def predict(n_clicks, latitude, longitude, bathrooms, bedrooms, square_feet, state, city, pets, source, category, photos):
    if n_clicks > 0:
        try:
            # Asignar 1 a la fuente seleccionada y 0 a las demás
            source_dict = {key: 1 if key == source else 0 for key in source_sites.keys()}
            
            # Asignar categoría seleccionada
            category_dict = {
                "category_housing/rent/home": 1 if category == 'home' else 0,
                "category_housing/rent/short_term": 1 if category == 'short_term' else 0
            }
            
            # Crear el DataFrame con los valores de entrada
            ejemplo = pd.DataFrame({
                "latitude": [latitude],
                "longitude": [longitude],
                "bathrooms": [bathrooms],
                "bedrooms": [bedrooms],
                "square_feet": [square_feet],
                "cityname": [city],
                "state": [state],
                'amenities_count': [1],
                'pets_allowed_Cats,Dogs': [pets],
                'pets_allowed_No': [1 - pets],
                **source_dict,
                **category_dict,
                "has_photo_Thumbnail": [photos],
                "has_photo_Yes": [photos]
            })
            
            ejemplo_scaled = X_scaler3.transform(ejemplo)
            ejemplo_polinomico = poly.transform(ejemplo_scaled)
            y_predicho_scaled = intercepto + np.dot(ejemplo_polinomico, coeficientes)
            y_predicho_real = y_scaler3.inverse_transform(y_predicho_scaled.reshape(-1, 1))
            
            return f"El precio estimado del apartamento es: {y_predicho_real[0][0]:.2f}"
        except Exception as e:
            return f"Error en la predicción: {str(e)}"
    return "Ingrese los datos y presione 'Predecir'"

if __name__ == '__main__':
    app.run_server(debug=True)













