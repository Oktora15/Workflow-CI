import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================
# SET EXPERIMENT
# ======================
mlflow.set_experiment("Heart_Disease_Experiment")

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAINING + LOGGING
# ======================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # log parameter
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # log metric
    mlflow.log_metric("accuracy", acc)

    # log model ke MLflow (tracking)
    mlflow.sklearn.log_model(model, "model")

    # ======================
    # SAVE MODEL LOKAL
    # ======================
    local_model_dir = "model"
    mlflow.sklearn.save_model(model, local_model_dir)

print("Training selesai + model tersimpan.")