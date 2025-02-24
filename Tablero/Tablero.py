import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Cargar coeficientes del modelo con manejo de errores
try:
    coeficientes = np.load("coeficientes_modelo.npy")
    intercepto = np.load("intercepto_modelo.npy")
except FileNotFoundError:
    raise FileNotFoundError("Error: No se encuentran los archivos de coeficientes. Asegúrate de guardarlos correctamente.")

# Cargar los datos limpios para ajustar el escalador
try:
    df = pd.read_csv("datos_limpios.csv", encoding="latin-1", sep=";")
except FileNotFoundError:
    raise FileNotFoundError("Error: No se encuentra el archivo 'datos_limpios.csv'.")

# Definir las variables utilizadas en el modelo
variables = [
    "amenities_count", "category_housing/rent/home", "category_housing/rent/short_term", "has_photo_Thumbnail",
    "has_photo_Yes", "pets_allowed_Cats,Dogs", "pets_allowed_Dogs", "pets_allowed_No", 'source_Listanza',
    'source_ListedBuy', 'source_RENTOCULAR', 'source_Real Estate Agent', 'source_RentDigs.com',
    'source_RentLingo', 'source_rentbits', 'source_tenantcloud'
]

# Verificar número de variables antes de transformación
print(f"Número de variables de entrada: {len(variables)}")

# Asegurar que todas las variables necesarias están en el DataFrame
missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    raise ValueError(f"Las siguientes variables están ausentes en el archivo CSV: {missing_vars}")

# Convertir columnas numéricas con comas a float
def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=True).astype(float)
    return df

df = convert_to_numeric(df, [col for col in variables if df[col].dtype == 'object'])

# Seleccionar variables
X1 =["latitude","longitude","bathrooms","bedrooms","square_feet","cityname","state","amenities_count","category","source","pets_allowed"]

# Ajustar escaladores con los datos originales
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X1)
print(X1)

# Forzar el uso de grado 3 ya que se confirmó en el entrenamiento
grado_polinomio = 3

# Aplicar PolynomialFeatures con include_bias=False para evitar el término de intercepción adicional
poly = PolynomialFeatures(degree=grado_polinomio, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Diagnóstico: Comparar dimensiones con coeficientes
print(f"Número de características generadas por PolynomialFeatures: {X_poly.shape[1]}")
print(f"Número de coeficientes esperados por el modelo: {coeficientes.shape[0]}")

# Guardar nombres de características generadas
nombres_features = poly.get_feature_names_out()

# Guardar los nombres de las variables en un archivo CSV
pd.DataFrame(nombres_features, columns=["Feature Names"]).to_csv("nombres_variables_generadas.csv", index=False)

# Verificar dimensiones
if coeficientes.shape[0] != X_poly.shape[1]:
    raise ValueError(f"Error de dimensiones: coeficientes ({coeficientes.shape[0]}) vs características ({X_poly.shape[1]}). Verifica que el grado del polinomio sea el correcto y que las variables sean las mismas.")

# Función para calcular el precio estimado
def calcular_precio(*args):
    X_input = pd.DataFrame([args], columns=variables)
    X_input_scaled = scaler_X.transform(X_input)
    X_input_poly = poly.transform(X_input_scaled)
    
    if X_input_poly.shape[1] != coeficientes.shape[0]:
        raise ValueError(f"Error de dimensiones en la predicción: coeficientes ({coeficientes.shape[0]}) vs entrada polinómica ({X_input_poly.shape[1]}).")
    
    precio_estimado = np.dot(X_input_poly, coeficientes) + intercepto
    return precio_estimado[0]

# Prueba con valores de entrada
test_input = (4, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0)
precio_estimado = calcular_precio(*test_input)
print(f"Precio estimado del apartamento: ${precio_estimado:,.2f}")
