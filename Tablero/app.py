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
    html.H1("Predicción del precio de un apartamento"),
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
    html.Label("Seleccione el estado:"),
    dcc.Dropdown(id='input-state', options=state_options, value=state_options[0]['value']),
    html.Label("Seleccione la ciudad:"),
    dcc.Dropdown(id='input-city', options=city_options, value=city_options[0]['value']),
    html.Button('Predecir', id='predict-button', n_clicks=0),
    html.H3(id='prediction-output', style={'marginTop': '20px'})
])

# Callback para actualizar el estado seleccionado desde el mapa
@app.callback(
    Output("input-state", "value"),
    Input("state-map", "clickData")
)
def update_selected_state(click_data):
    if click_data:
        state_abbr = click_data['points'][0]['location']
        state_code = state_encoder.transform([state_abbr])[0]
        return state_code
    return state_options[0]['value']

# Callback para actualizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('input-latitude', 'value'),
    Input('input-longitude', 'value'),
    Input('input-bathrooms', 'value'),
    Input('input-bedrooms', 'value'),
    Input('input-state', 'value'),
    Input('input-city', 'value')
)
def predict(n_clicks, latitude, longitude, bathrooms, bedrooms, state, city):
    if n_clicks > 0:
        try:
            # Crear el DataFrame con los valores de entrada
            ejemplo = pd.DataFrame({
                "latitude": [latitude],
                "longitude": [longitude],
                "bathrooms": [bathrooms],
                "bedrooms": [bedrooms],
                "square_feet": [200],
                "cityname": [city],
                "state": [state],
                'amenities_count': [1],
                'pets_allowed_Cats,Dogs': [0],
                'pets_allowed_No': [1],
                'source_ListedBuy': [0],
                'source_RentDigs.com': [0],
                'source_RentLingo': [1],
                "source_Listanza": [0],
                "has_photo_Thumbnail": [1],
                "has_photo_Yes": [0],
                "category_housing/rent/home": [0],
                "category_housing/rent/short_term": [0]
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









