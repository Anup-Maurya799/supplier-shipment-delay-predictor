from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "delay_predictor.pkl"
)


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained model not found. "
            "Run train_model.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


def predict_delay(input_data):

    model = load_model()

    df = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(df)[0]

    probability = (
        model.predict_proba(df)[0][1]
    )

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }


def get_prediction_label(prediction):

    if prediction == 1:
        return "Delayed"

    return "On Time"