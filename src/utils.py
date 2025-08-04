# entrenar_modelo.py
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Datos de ejemplo (X: número, y: su cuadrado)
X = np.array([[i] for i in range(0, 101)])
y = X.flatten() ** 2

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Guardar el modelo entrenado en un archivo .pkl
with open("modelo.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo entrenado y guardado como 'modelo.pkl'")
