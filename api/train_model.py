import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def train_and_save_model():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), 'notebooks', 'housing', 'housing.csv')
    
    # 1. Cargar y verificar los datos
    try:
        print(f"Intentando cargar archivo desde: {csv_path}")
        # Intentar cargar desde data/
        df = pd.read_csv('data/housing.csv')
        print("Archivo cargado exitosamente.")
        
        # Mostrar información detallada del dataset
        print("\nColumnas disponibles en el dataset:")
        for col in df.columns:
            print(f"- {col} (tipo: {df[col].dtype})")
        
        print("\nPrimeras 5 filas del dataset:")
        print(df.head())
            
    except FileNotFoundError:
        try:
            # Intentar cargar desde la ruta relativa
            df = pd.read_csv('../notebooks/housing/housing.csv')
        except FileNotFoundError:
            print("Error: No se puede encontrar el archivo housing.csv")
            print("Ruta actual:", os.getcwd())
            return
    except Exception as e:
        print(f"Error al cargar el archivo: {str(e)}")
        return

    # 2. Análisis Exploratorio
    print("\nInformación del dataset:")
    print(df.info())
    
    # 3. Preparación del Modelo
    try:
        # Identificar la variable objetivo (precio)
        target_column = None
        possible_target_columns = ['median_house_value', 'price', 'value', 'house_value']
        
        for col in possible_target_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            print("\nError: No se encontró la columna objetivo (precio de la casa)")
            print("Columnas disponibles:", df.columns.tolist())
            return
            
        # Identificar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Excluir la variable objetivo
        features = [col for col in numeric_cols if col != target_column]
        
        print(f"\nVariable objetivo seleccionada: {target_column}")
        print("\nCaracterísticas seleccionadas para el modelo:")
        print(features)
        
        X = df[features]
        y = df[target_column]
        
        # Verificar que tenemos datos
        print("\nDimensiones de X:", X.shape)
        print("Dimensiones de y:", y.shape)
        
    except Exception as e:
        print(f"Error en la preparación de datos: {str(e)}")
        print("Detalles adicionales:")
        print("Columnas en el dataset:", df.columns.tolist())
        return

    # 4. Manejo de valores nulos
    X = X.fillna(X.mean())

    # 5. División de datos y entrenamiento
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluación
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nResultados del modelo:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        
    except Exception as e:
        print(f"Error en el entrenamiento del modelo: {str(e)}")
        return

    # 6. Guardar el modelo y metadatos
    try:
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(models_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(features, f)
            
        print("\nModelo guardado exitosamente en la carpeta 'models'")
        
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")
        return
    
    return features

if __name__ == "__main__":
    print("Iniciando entrenamiento del modelo...")
    features = train_and_save_model()
    if features:
        print("\nProceso completado exitosamente.")
    else:
        print("\nEl proceso no se completó debido a errores.") 