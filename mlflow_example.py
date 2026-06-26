import mlflow

EXPERIMENT_NAME = "count-numbers"
mlflow.set_tracking_uri("http://192.168.0.103:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active Experiment: {mlflow.get_experiment_by_name(EXPERIMENT_NAME)}")

with mlflow.start_run():
    mlflow.log_trace("Kitty", "What is the answer?", "I refuse to tell")
    mlflow.log_metrics({"accuracy": 1.0, "spontaneouity": 2.0})
    mlflow.log_param("test_param", "test_value")
    print("✓ Successfully connected to MLflow!")
