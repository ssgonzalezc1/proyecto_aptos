import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

# Load cleaned data
df = pd.read_csv("datos_limpios.csv", encoding="latin-1", sep=";")

# Load coefficients and intercept from .npy files
coeficientes = np.load("coeficientes_modelo.npy")
intercepto = np.load("intercepto_modelo.npy")[0]

# Define model features manually
model_features = ['amenities_count', 'category_housing/rent/home', 'category_housing/rent/short_term',
                  'has_photo_Thumbnail', 'has_photo_Yes', 'pets_allowed_Cats,Dogs', 'pets_allowed_Dogs',
                  'pets_allowed_No', 'source_Listanza', 'source_ListedBuy', 'source_RENTOCULAR',
                  'source_Real Estate Agent', 'source_RentDigs.com', 'source_RentLingo',
                  'source_rentbits', 'source_tenantcloud']

# Define Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Polynomial Regression: Apartment Rental Prices"),
    html.Div([
        html.Label("Select Amenities Count:"),
        dcc.Slider(
            id='amenities-slider',
            min=df['amenities_count'].min(),
            max=df['amenities_count'].max(),
            step=1,
            value=df['amenities_count'].median()
        ),

        html.Label("Select Housing Category:"),
        dcc.Dropdown(
            id='category-dropdown',
            options=[
                {'label': 'Home', 'value': 'category_housing/rent/home'},
                {'label': 'Short Term', 'value': 'category_housing/rent/short_term'}
            ],
            value='category_housing/rent/home'
        ),

        html.Label("Photo Availability:"),
        dcc.RadioItems(
            id='photo-radio',
            options=[
                {'label': 'Thumbnail', 'value': 'has_photo_Thumbnail'},
                {'label': 'Yes', 'value': 'has_photo_Yes'}
            ],
            value='has_photo_Yes'
        ),

        html.Label("Pets Allowed:"),
        dcc.Dropdown(
            id='pets-dropdown',
            options=[
                {'label': 'Cats & Dogs', 'value': 'pets_allowed_Cats,Dogs'},
                {'label': 'Dogs', 'value': 'pets_allowed_Dogs'},
                {'label': 'No Pets', 'value': 'pets_allowed_No'}
            ],
            value='pets_allowed_No'
        ),

        html.Label("Select Source:"),
        dcc.Dropdown(
            id='source-dropdown',
            options=[
                {'label': src, 'value': src} for src in model_features[8:]
            ],
            value='source_Listanza'
        ),

        html.Button('Predict Price', id='predict-button', n_clicks=0)
    ]),

    html.Div(id='prediction-output', style={'margin-top': '20px'}),
    dcc.Graph(id='regression-graph')
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('regression-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [dash.State('amenities-slider', 'value'),
     dash.State('category-dropdown', 'value'),
     dash.State('photo-radio', 'value'),
     dash.State('pets-dropdown', 'value'),
     dash.State('source-dropdown', 'value')]
)
def update_output(n_clicks, amenities, category, photo, pets, source):
    if n_clicks == 0:
        return "", px.scatter()
    
    # Create feature vector manually
    input_data = {feature: 0 for feature in model_features}
    input_data['amenities_count'] = amenities
    input_data[category] = 1
    input_data[photo] = 1
    input_data[pets] = 1
    input_data[source] = 1
    
    # Convert dictionary to array in correct order
    input_vector = np.array([input_data[feature] for feature in model_features])
    
    # Calculate prediction manually
    predicted_price = np.dot(coeficientes, input_vector) + intercepto
    
    return f"Predicted Rent Price: ${predicted_price:.2f}", px.scatter()

if __name__ == '__main__':
    app.run_server(debug=True)


