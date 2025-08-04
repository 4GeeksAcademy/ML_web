# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Cargar el modelo entrenado
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# Crear la app Flask
app = Flask(__name__)

# Ruta para predecir el cuadrado
@app.route("/predecir", methods=["GET"])
def predecir():
    numero = request.args.get("numero", type=float)
    
    if numero is None:
        return jsonify({"error": "Falta el parámetro 'numero'"})

    pred = modelo.predict(np.array([[numero]]))[0]
    return jsonify({
        "input": numero,
        "cuadrado_estimado": pred
    })

# Ejecutar la app
if __name__ == "__main__":
    app.run(debug=True)
