from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar modelo entrenado
model = tf.keras.models.load_model("modelo_credito.h5")

# Simulación del scaler (debes guardar el scaler de entrenamiento real)
scaler = StandardScaler()
scaler.mean_ = np.array([130000, 65000, 1.6, 28000])  # Simulación
scaler.scale_ = np.array([50000, 20000, 0.5, 10000])  # Simulación

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[data["ingresos"], data["inventario"], data["razon_corriente"], data["deuda"]]])
    features_scaled = scaler.transform(features)
    probabilidad = model.predict(features_scaled)[0][0]
    
    umbral = 0.1  # Ajusta según la estrategia de riesgo
    decision = "Otorgar crédito" if probabilidad < umbral else "Rechazar crédito"
    
    return jsonify({"probabilidad": float(probabilidad), "decision": decision})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
