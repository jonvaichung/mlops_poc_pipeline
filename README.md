# Iris MLOps Pipeline POC

A lightweight, end-to-end MLOps pipeline demonstrating model training, tracking via MLflow Model Registry, a FastAPI serving layer, and production-ready containerisation with Docker.

---

## Tech Stack
* **Machine Learning:** Scikit-Learn (Iris Dataset)
* **Model Registry & Tracking:** MLflow (SQLite backend)
* **API Framework:** FastAPI & Pydantic
* **Containerisation:** Docker

---

## Repo Structure

```text
mlops_poc_pipeline/
├── conf/               # Configuration files (hyperparameters, paths, etc.)
├── app.py              # FastAPI application & serving logic
├── train.py            # Model training & experiment tracking script
├── register_model.py   # MLflow model registration script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build instructions
├── mlflow.db           # SQLite database (generated local MLflow registry)
└── mlruns/             # MLflow artifact storage (generated local model files)
```

## Quick Start

### 1. Prerequisites
Ensure you have **Docker** installed and running in your enviroment.

### 2. Build Docker Image
```bash
docker build -t iris-mlops-app .
```

### 3. Run the Container
Because MLflow tracks absolute paths for model artifacts, run the container using a volume mount to align your host paths with the container environment:
```bash
docker run -d -p 8000:8000 \
  -v $(pwd):/workspaces/mlops_poc_pipeline \
  -w /workspaces/mlops_poc_pipeline \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  --name iris-container iris-mlops-app
```

## API Usage
Once the container is running, access the Swagger API to test predictions through the forwarded port.

### Example Request (`POST /predict`)
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example Response
```json
{
  "prediction_class": "setosa",
  "prediction_id": 0
}
```

## Useful Docker Commands
* View live logs: `docker logs -f iris-container`
* Stop container:  `docker stop iris-container`
* Remove container: `docker rm iris-container`