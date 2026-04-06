from model import load_model
from explain_lime import run_lime
from predict import predict
from visualization import create_dashboard
from mlflow_tracker import log_experiment

model = load_model()

image_path = "images/test.jpg"

print("Running prediction...")
predict(model, image_path)

print("Generating explanation...")
run_lime(model, image_path)

print("Creating visualization dashboard...")
create_dashboard()

print("Logging experiment with MLflow...")
log_experiment()

print("All tasks completed.")
