import numpy as np
import pandas as pd
import pickle

# Cargar el modelo entrenado
with open("modelo_polynomial.pkl", "rb") as f:
    coeficientes, intercepto, poly, X_scaler3,y_scaler3, nombres_features = pickle.load(f)

ejemplo = pd.DataFrame({"latitude": [33.5178], "longitude": [-112.085],"bathrooms":[1],"bedrooms":[1],"square_feet":[200],"cityname":[1100],"state":[3],'amenities_count':[1], 'pets_allowed_Cats,Dogs':[0], 'pets_allowed_No':[1], 'source_ListedBuy':[0], 'source_RentDigs.com':[0], 'source_RentLingo':[1],"source_Listanza":[0],"has_photo_Thumbnail":[1],"has_photo_Yes":[0],"category_housing/rent/home":[0],"category_housing/rent/short_term":[0]
})
ejemplo_scaled = X_scaler3.transform(ejemplo)
ejemplo_polinomico = poly.transform(ejemplo_scaled)
y_predicho_scaled = intercepto + np.dot(ejemplo_polinomico, coeficientes)
y_predicho_real = y_scaler3.inverse_transform(y_predicho_scaled.reshape(-1, 1))
print(f"El valor predicho de y es: {y_predicho_real[0][0]}")



