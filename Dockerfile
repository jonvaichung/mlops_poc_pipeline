# Official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variable for MLflow tracking URI so it looks at the local sqlite db inside the container
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Copy dependency definitions and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . /app

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]