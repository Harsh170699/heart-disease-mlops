# Heart Disease Prediction — MLOps Project

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for binary heart disease classification using the UCI Cleveland Heart Disease dataset.

The project covers data preparation, preprocessing, model training and evaluation, experiment tracking with MLflow, model persistence, automated testing, REST API development, Docker containerization, and continuous integration using GitHub Actions.

**Disclaimer:** This is an academic demonstration and is not intended for clinical use or medical diagnosis.

## 2. Objectives

- Build a reproducible machine learning pipeline.
- Compare Logistic Regression and Random Forest classifiers.
- Track experiments using MLflow.
- Save and reuse the trained model.
- Expose predictions through a FastAPI REST API.
- Containerize the application using Docker.
- Validate functionality using automated tests.
- Configure a GitHub Actions CI workflow.

## 3. Dataset

- **Source:** UCI Cleveland Heart Disease dataset
- **Dataset file:** `processed.cleveland.data`
- **Original observations:** 303
- **Input features:** 13
- **Prediction target:** Binary classification

The original target variable is converted into:
- `0`: No heart disease indicated by the dataset target
- `1`: Heart disease indicated by the dataset target

Missing values are handled through the preprocessing pipeline.

## 4. Machine Learning Pipeline

1. Load and inspect the dataset.
2. Clean the data and handle missing values.
3. Convert the original target into a binary target.
4. Split the data into training and testing sets using stratification.
5. Apply preprocessing:
   - Numerical features: median imputation and standardization.
   - Categorical features: most-frequent imputation and one-hot encoding.
6. Train and compare Logistic Regression and Random Forest.
7. Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Track experiments with MLflow.
9. Save the selected model pipeline.
10. Reuse the saved pipeline for predictions.

## 5. Model Evaluation

The following are the recorded test-set results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8852 | 0.8852 |
| Precision | 0.8387 | 0.8387 |
| Recall | 0.9286 | 0.9286 |
| F1-score | 0.8814 | 0.8814 |
| ROC-AUC | 0.9665 | 0.9443 |

Logistic Regression was selected as the saved model based on the project evaluation.

These results are from this particular train/test split and should not be interpreted as clinical performance guarantees.

## 6. Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- MLflow
- joblib
- FastAPI
- Pydantic
- pytest
- Docker and Docker Compose
- GitHub Actions

## 7. Project Structure

```text
heart_disease_mlops/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── best_heart_disease_model.joblib
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── predict.py
├── tests/
│   ├── test_api.py
│   └── test_predict.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-lock.txt
└── requirements-api.txt
```

Additional directories and files may be present in the working project.

## 8. Run the API Locally

Install the API dependencies:

```bash
pip install -r requirements-api.txt
```

Start the API from the project root:

```bash
uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Root endpoint: http://127.0.0.1:8000/

The prediction endpoint is `POST /predict`.

## 9. Run Automated Tests

From the project root:

```bash
python -m pytest tests/ -v
```

The current local test suite contains eight tests covering the prediction module, API endpoints, output structure, and invalid inputs.

## 10. Docker Deployment

Build the image from the project root:

```bash
docker build -t heart-disease-api:1.0 .
```

Start the container:

```bash
docker run -d --name heart-disease-api-container -p 8000:8000 --restart unless-stopped heart-disease-api:1.0
```

Alternatively, use Docker Compose:

```bash
docker compose up -d
```

Check container status:

```bash
docker ps
```

View container logs:

```bash
docker logs heart-disease-api-container
```

Stop the container:

```bash
docker compose down
```

## 11. Continuous Integration

The GitHub Actions workflow is located at `.github/workflows/ci.yml`.

It is configured to:
- Run on pushes and pull requests targeting `main` or `master`.
- Support manual execution using `workflow_dispatch`.
- Set up Python 3.11.
- Install API dependencies and test dependencies.
- Verify that the trained model artifact exists.
- Run the automated test suite.

The workflow must be pushed to a GitHub repository before its hosted CI execution can be verified.

## 12. Limitations

- The dataset is relatively small and is used for academic demonstration.
- Model metrics depend on the chosen data split and evaluation procedure.
- The API does not replace professional medical assessment.
- The current CI workflow tests the application but does not publish a Docker image or deploy it to a cloud service.

## 13. Conclusion

This project demonstrates the integration of machine learning development with practical MLOps components, including experiment tracking, model persistence, API serving, containerization, automated testing, and CI configuration.
