import mlflow
import json

def log_experiment():

    mlflow.set_experiment("ExplainAI-Pro")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model", "ResNet18")
        mlflow.log_param("explanation_method", "LIME")

        # Log prediction result
        with open("../outputs/prediction.json") as f:
            prediction = json.load(f)

        mlflow.log_metric("confidence", prediction["confidence"])

        # Log artifacts
        mlflow.log_artifact("outputs/lime_result.png")
        mlflow.log_artifact("outputs/comparison.png")

        print("Experiment logged to MLflow.")
