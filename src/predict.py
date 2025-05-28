from .model_loader import MODELS, scaler
import joblib
import numpy as np
import pandas as pd
from src.feature_names import FEATURE_NAMES  # Imported dynamic feature list

imputer = joblib.load("models/imputer.pkl")

def make_prediction(features: dict[str, float], model_name: str):
    # Verify that all expected features are provided
    missing_features = [f for f in FEATURE_NAMES if f not in features]
    extra_features = [f for f in features if f not in FEATURE_NAMES]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    if extra_features:
        raise ValueError(f"Unexpected features: {extra_features}")

    model = MODELS.get(model_name.lower())
    if not model:
        raise ValueError("Model not found")

    # Create DataFrame with one row using the expected feature order
    df = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)

    try:
        # Apply imputation and scaling as needed
        features_imputed = imputer.transform(df)
        features_scaled = scaler.transform(features_imputed)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        classes = model.classes_

        return {
            "predicted_class": prediction,
            "probabilities": dict(zip(classes, probabilities))
        }

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"error": str(e)}
