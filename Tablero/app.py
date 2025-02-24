import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import pickle
import dash_bootstrap_components as dbc

# Cargar el modelo entrenado
with open("modelo_polynomial.pkl", "rb") as f:
    coeficientes, intercepto, poly, X_scaler3, y_scaler3, nombres_features = pickle.load(f)

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Diseño de la aplicación
app.layout = dbc.Container([
    html.H1("Predicción de Valor con Modelo Polynomial"),
    html.Label("Ingrese la latitud:"),
    dcc.Input(id='input-latitude', type='number', value=0, step=0.01),
    html.Button('Predecir', id='predict-button', n_clicks=0),
    html.H3(id='prediction-output', style={'marginTop': '20px'})
])

# Callback para actualizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('input-latitude', 'value')
)
def predict(n_clicks, latitude):
    if n_clicks > 0:
        try:
            # Crear el DataFrame con los valores de entrada
            ejemplo = pd.DataFrame({
                "latitude": [latitude],
                "longitude": [-112.085],
                "bathrooms": [2],
                "bedrooms": [1],
                "square_feet": [200],
                "cityname": [1100],
                "state": [3],
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
            
            # Transformar los datos con los escaladores y el modelo
            ejemplo_scaled = X_scaler3.transform(ejemplo)
            ejemplo_polinomico = poly.transform(ejemplo_scaled)
            y_predicho_scaled = intercepto + np.dot(ejemplo_polinomico, coeficientes)
            y_predicho_real = y_scaler3.inverse_transform(y_predicho_scaled.reshape(-1, 1))
            
            return f"El valor predicho de y es: {y_predicho_real[0][0]:.2f}"
        except Exception as e:
            return f"Error en la predicción: {str(e)}"
    return "Ingrese la latitud y presione 'Predecir'"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)




