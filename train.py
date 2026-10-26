import hydra
from omegaconf import DictConfig
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("--- Starting Training Pipeline ---")

    # 1. Load Dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=cfg.training.test_size, 
        random_state=cfg.model.random_state
    )

    # 2. Set up MLflow Tracking (using a local SQLite database)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_experiment")

    with mlflow.start_run():
        # Log Hydra configuration parameters
        mlflow.log_params(cfg.model)
        mlflow.log_param("test_size", cfg.training.test_size)

        # 3. Train Model
        model = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state
        )
        model.fit(X_train, y_train)

        # 4. Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # 5. Log model artifact (with skops trusted types for tree-based models)
        mlflow.sklearn.log_model(
            model, 
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("Model logged to MLflow successfully!")

if __name__ == "__main__":
    main()