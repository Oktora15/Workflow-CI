import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ======================
# SET EXPERIMENT
# ======================
#mlflow.set_experiment("Heart_Disease_Experiment")

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

    # MODEL
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # ======================
    # LOG PARAMETER
    # ======================
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # ======================
    # LOG METRIC
    # ======================
    mlflow.log_metric("accuracy", acc)

    # ======================
    # LOG MODEL (MLFLOW TRACKING)
    # ======================
    mlflow.sklearn.log_model(model, "model")

    # ======================
    # SAVE MODEL LOKAL (UNTUK INFERENCE)
    # ======================
    local_model_dir = "model"

    # HAPUS FOLDER LAMA JIKA ADA
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)

    # SIMPAN MODEL BARU
    mlflow.sklearn.save_model(model, local_model_dir)

print("Training selesai + model berhasil disimpan.")