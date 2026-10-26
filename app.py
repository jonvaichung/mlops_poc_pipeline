from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="Iris MLOps MVP API", description="Serves the Production Iris model from MLflow Registry")

# Load the model from the MLflow Model Registry (Production stage)
MODEL_NAME = "IrisRandomForest"
MODEL_URI = f"models:/{MODEL_NAME}/Production"

print(f"Loading model from registry: {MODEL_URI}...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model from registry yet ({e}). Make sure train.py and register_model.py have been run.")
    model = None

# Define the expected input data structure using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Welcome to the MLOps MVP API. Use the /docs endpoint to test predictions."}

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Format input for scikit-learn
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    # Make prediction
    prediction = model.predict(features)
    
    # Map numerical prediction back to Iris class names
    class_names = ["setosa", "versicolor", "virginica"]
    predicted_class = class_names[int(prediction[0])]
    
    return {
        "prediction_class": predicted_class,
        "prediction_id": int(prediction[0])
    }