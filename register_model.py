import mlflow
from mlflow.tracking import MlflowClient

def main():
    # 1. Connect to the local MLflow tracking database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    # 2. Find the most recent training run in 'iris_experiment'
    experiment = client.get_experiment_by_name("iris_experiment")
    if not experiment:
        print("Error: 'iris_experiment' not found. Run train.py first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Error: No training runs found.")
        return

    latest_run_id = runs[0].info.run_id
    print(f"Found Latest Run ID: {latest_run_id}")

    # 3. Register the model artifact into the MLflow Model Registry
    model_name = "IrisRandomForest"
    model_uri = f"runs:/{latest_run_id}/model"
    
    print(f"Registering model to registry as '{model_name}'...")
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version
    print(f"Successfully registered Version {version}")

    # 4. Programmatically manage stage transitions
    # Step A: Move to Staging
    print(f"Transitioning Version {version} to Staging...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )

    # Step B: Promote to Production
    print(f"Promoting Version {version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print("--- Model Lifecycle Management Complete: Now in Production ---")

if __name__ == "__main__":
    main()