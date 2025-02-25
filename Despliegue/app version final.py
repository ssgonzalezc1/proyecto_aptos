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

df = df.dropna(subset=['state', 'cityname'])

state_encoder = LabelEncoder()
df['state_encoded'] = state_encoder.fit_transform(df['state'])
city_encoder = LabelEncoder()
df['city_encoded'] = city_encoder.fit_transform(df['cityname'])

state_options = [{'label': state, 'value': code} for state, code in zip(state_encoder.classes_, state_encoder.transform(state_encoder.classes_))]
city_options = [{'label': city, 'value': code} for city, code in zip(city_encoder.classes_, city_encoder.transform(city_encoder.classes_))]

source_options = [
    {'label': 'ListedBuy', 'value': 'source_ListedBuy'},
    {'label': 'RentDigs.com', 'value': 'source_RentDigs.com'},
    {'label': 'RentLingo', 'value': 'source_RentLingo'},
    {'label': 'Listanza', 'value': 'source_Listanza'}
]

fig = px.choropleth(
    df,
    locations="state",
    locationmode="USA-states",
    scope="usa",
    title="Seleccione un estado en el mapa"
)
fig.update_layout(coloraxis_showscale=False)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    html.Label("Ingrese el tamaño en pies cuadrados:"),
    dcc.Input(id='input-square-feet', type='number', value=200, step=10),
    html.Label("Seleccione el estado:"),
    dcc.Dropdown(id='input-state', options=state_options, value=state_options[0]['value']),
    html.Label("Seleccione la ciudad:"),
    dcc.Dropdown(id='input-city', options=city_options, value=city_options[0]['value']),
    html.Label("¿Se permiten mascotas?"),
    dcc.RadioItems(
        id='input-pets',
        options=[
            {'label': 'Sí', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Seleccione la fuente:"),
    dcc.Dropdown(id='input-source', options=source_options, value='source_RentLingo'),
    html.Button('Predecir', id='predict-button', n_clicks=0),
    html.H3(id='prediction-output', style={'marginTop': '20px'})
])

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
    Input('input-source', 'value')
)
def predict(n_clicks, latitude, longitude, bathrooms, bedrooms, square_feet, state, city, pets, source):
    if n_clicks > 0:
        try:
            pets_allowed_cats_dogs = 1 if pets == 'yes' else 0
            pets_allowed_no = 0 if pets == 'yes' else 1

            source_values = {s['value']: 1 if s['value'] == source else 0 for s in source_options}

            ejemplo = pd.DataFrame({
                "latitude": [latitude],
                "longitude": [longitude],
                "bathrooms": [bathrooms],
                "bedrooms": [bedrooms],
                "square_feet": [square_feet],
                "cityname": [city],
                "state": [state],
                'amenities_count': [1],
                'pets_allowed_Cats,Dogs': [pets_allowed_cats_dogs],
                'pets_allowed_No': [pets_allowed_no],
                **source_values,
                "has_photo_Thumbnail": [1],
                "has_photo_Yes": [1],
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















