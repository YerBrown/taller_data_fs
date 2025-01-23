import pandas as pd
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import plotly.express as px
import numpy as np
from pathlib import Path
import pickle

app = Flask(__name__)
cors = CORS(app)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Housing Price Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .success {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Housing Price Prediction</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label for="surface">Surface (m²):</label>
                <input type="number" id="surface" name="surface" required>
            </div>
            <div class="form-group">
                <label for="bedrooms">Number of Bedrooms:</label>
                <input type="number" id="bedrooms" name="bedrooms" required>
            </div>
            <div class="form-group">
                <label for="restrooms">Number of Restrooms:</label>
                <input type="number" id="restrooms" name="restrooms" required>
            </div>
            <button type="submit">Predict Price</button>
        </form>
        <div id="result" class="result" style="display: none;"></div>
        <div id="plotly-graph"></div>
    </div>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                surface: parseInt(document.getElementById('surface').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                restrooms: parseInt(document.getElementById('restrooms').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();
                
                const resultDiv = document.getElementById('result');
                if (data.error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `Error: ${data.error}`;
                } else {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `Predicted Price: ${parseFloat(data.prediction[0]).toFixed(2)}€`;
                    
                    // Display the graph
                    const graphData = JSON.parse(data.graph);
                    Plotly.newPlot('plotly-graph', graphData.data, graphData.layout);
                }
                resultDiv.style.display = 'block';
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `Error: ${error.message}`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

# Load the model (will be created by the notebook)
print("Attempting to load model...")
try:
    # Get the absolute path to the model file
    model_path = Path(__file__).parent / 'models' / 'model.pkl'
    print(f"Looking for model at: {model_path}")
    
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        model_data = None
    else:
        print("Model file found, loading...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_data = None

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
    data = request.get_json()

    try:
        # Get base features
        surface = parse_int(data['surface'])
        bedrooms = parse_int(data['bedrooms'])
        restrooms = parse_int(data['restrooms'])
        
        # Calculate additional features
        rooms_total = bedrooms + restrooms
        avg_room_size = surface / rooms_total
        
        # Create input DataFrame with all features
        input_data = pd.DataFrame({
            'surface': [surface],
            'bedrooms': [bedrooms],
            'restrooms': [restrooms],
            'rooms_total': [rooms_total],
            'avg_room_size': [avg_room_size]
        })
        
        # Scale the input data
        input_scaled = model_data['scaler'].transform(input_data)
        
        # Make prediction
        prediction = parse_float(model_data['model'].predict(input_scaled))
        
        # Create visualization
        fig = px.bar(x=['Predicted Price'], y=prediction)
        fig.update_layout(
            title='Housing Price Prediction',
            xaxis_title="",
            yaxis_title="Price (€)",
            showlegend=False
        )
        graph_json = fig.to_json()

        return jsonify({
            'prediction': prediction,
            'graph': graph_json
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def parse_int(x):
    # Handle both single values and lists
    if isinstance(x, list):
        return [int(i) for i in x]
    return int(x)
    
def parse_float(x):
    # Handle numpy arrays, pandas series, lists and single values
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return [float(i) for i in x]
    return float(x)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
