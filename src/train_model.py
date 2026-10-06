from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smart_logistics_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

sys.path.append(str(PROJECT_ROOT))
from src.preprocessing import load_data, clean_data, create_features, get_model_data

print("\n" + "=" * 70)
print("AI SHIPMENT DELAY PREDICTOR - MODEL TRAINING")
print("=" * 70)

# 1. Load + clean
_df = load_data(DATA_PATH)
_df = clean_data(_df)
_df = create_features(_df)
X, y = get_model_data(_df)

print(f"Dataset shape: {X.shape}")
print("\nFeatures used by the model:")
for feature in X.columns:
    print(" -", feature)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")
    return model, metrics

logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000)),
])

random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )),
])

logistic_model, logistic_metrics = evaluate("LOGISTIC REGRESSION", logistic)
rf_model, rf_metrics = evaluate("RANDOM FOREST", random_forest)

# Select by F1 only. This is a documented metric-based selection, not a
# claim that the dataset is highly predictive.
if rf_metrics["f1"] >= logistic_metrics["f1"]:
    best_model = rf_model
    best_name = "Random Forest"
    best_metrics = rf_metrics
else:
    best_model = logistic_model
    best_name = "Logistic Regression"
    best_metrics = logistic_metrics

joblib.dump(best_model, MODEL_DIR / "delay_predictor.pkl")

model_info = {
    "model_name": best_name,
    "features": X.columns.tolist(),
    "metrics": best_metrics,
    "target": "Logistics_Delay",
    "excluded_features": ["Shipment_Status", "Traffic_Status", "Logistics_Delay_Reason"],
}
joblib.dump(model_info, MODEL_DIR / "model_info.pkl")

print("\n" + "=" * 70)
print(f"Saved model: {best_name}")
print(f"Model file : {MODEL_DIR / 'delay_predictor.pkl'}")
print("=" * 70)
