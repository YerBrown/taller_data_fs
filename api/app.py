from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Cargar el modelo y metadatos
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    model = pickle.load(open(os.path.join(models_dir, 'model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'), 'rb'))
    FEATURES = pickle.load(open(os.path.join(models_dir, 'features.pkl'), 'rb'))
    
    print("Modelo cargado exitosamente")
    print("Características requeridas:", FEATURES)
    
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    print("Asegúrate de haber ejecutado train_model.py primero")

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'ok',
        'message': 'House Price Prediction API',
        'endpoints': {
            '/': 'Esta información',
            '/features': 'Lista de características necesarias',
            '/predict': 'Realizar predicción (POST)',
            '/metadata': 'Información del modelo'
        }
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint para obtener la lista de características necesarias"""
    return jsonify({
        'features': FEATURES.tolist() if hasattr(FEATURES, 'tolist') else FEATURES,
        'example_values': {
            'id': 1,
            'surface': 100,
            'bedrooms': 3,
            'restrooms': 2,
            'rent': 1000,
            'is_province': True
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        # Verificar si hay datos en la solicitud
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos',
                'required_features': FEATURES
            }), 400

        data = request.json
        print("Datos recibidos:", data)  # Para debugging
        
        # Verificar que todas las características requeridas estén presentes
        missing_features = [feat for feat in FEATURES if feat not in data]
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Faltan las siguientes características: {missing_features}',
                'required_features': FEATURES
            }), 400
        
        # Preparar datos para la predicción
        try:
            features_array = np.array([[float(data[feature]) for feature in FEATURES]])
            print("Features array:", features_array)  # Para debugging
        except ValueError as ve:
            return jsonify({
                'success': False,
                'error': 'Error en el formato de los datos numéricos',
                'message': str(ve)
            }), 400
        
        # Realizar predicción
        prediction = model.predict(features_array)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'input_features': data
        })
    
    except Exception as e:
        print(f"Error en predicción: {str(e)}")  # Para debugging
        return jsonify({
            'success': False,
            'error': 'Error al realizar la predicción',
            'message': str(e)
        }), 500

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """Endpoint para obtener información sobre el modelo"""
    return jsonify({
        'model_type': 'Linear Regression',
        'features': FEATURES,
        'target': 'house_price',
        'api_version': '1.0'
    })

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    print("La API estará disponible en: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

