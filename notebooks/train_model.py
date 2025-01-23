import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from pathlib import Path

def train_and_save_model():
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        
        print("Loading data...")
        data_path = script_dir / 'housing.csv'
        print(f"Looking for data in: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        df = pd.read_csv(data_path)
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nSample of the data:")
        print(df.head())
        
        # Feature engineering and selection
        features = ['surface', 'bedrooms', 'restrooms']
        target = 'price'
        
        # Verify columns exist
        missing_columns = [col for col in features + [target] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in dataset: {missing_columns}")
        
        print("\nCleaning data...")
        # Remove outliers using IQR method
        def remove_outliers(df, columns):
            df_clean = df.copy()
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            return df_clean
        
        # Remove outliers from features and target
        df_clean = remove_outliers(df, features + [target])
        
        # Additional feature engineering
        df_clean['rooms_total'] = df_clean['bedrooms'] + df_clean['restrooms']
        df_clean['avg_room_size'] = df_clean['surface'] / df_clean['rooms_total']
        
        # Update features list with new features
        features = ['surface', 'bedrooms', 'restrooms', 'rooms_total', 'avg_room_size']
        
        # Split features and target
        X = df_clean[features]
        y = df_clean[target]
        
        print(f"\nTotal samples after cleaning: {len(df_clean)}")
        print(f"Features used: {features}")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\nTraining model...")
        # Initialize and train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Average CV R² score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Train final model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print("\nModel Performance:")
        print(f"Training MSE: {train_mse:.2f}")
        print(f"Test MSE: {test_mse:.2f}")
        print(f"Training R²: {train_r2:.2f}")
        print(f"Test R²: {test_r2:.2f}")
        print(f"Training MAE: {train_mae:.2f}€")
        print(f"Test MAE: {test_mae:.2f}€")
        
        print("\nFeature Importances:")
        importances = model.feature_importances_
        for feature, importance in zip(features, importances):
            print(f"{feature}: {importance:.4f}")
        
        # Save both the model and the scaler
        print("\nSaving model and scaler...")
        models_dir = script_dir.parent / 'api' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': features
        }
        
        model_path = models_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model and scaler saved successfully to {model_path}!")
        
        # Verify the model was saved
        if not model_path.exists():
            raise FileNotFoundError(f"Model file was not created at {model_path}")
            
        # Try to load the model to verify it was saved correctly
        with open(model_path, 'rb') as f:
            test_model_data = pickle.load(f)
        print("Model verified successfully!")
        
        # Print example prediction
        print("\nExample prediction:")
        example = pd.DataFrame({
            'surface': [100],
            'bedrooms': [2],
            'restrooms': [1],
            'rooms_total': [3],
            'avg_room_size': [33.33]
        })
        example_scaled = scaler.transform(example)
        pred = model.predict(example_scaled)[0]
        print(f"Price for 100m², 2 bedrooms, 1 bathroom: {pred:.2f}€")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 