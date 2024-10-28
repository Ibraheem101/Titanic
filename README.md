Under Construction ..

# Titanic Survival Prediction: Dockerized Machine Learning Model Deployment
## Overview
This project demonstrates the deployment of a machine learning model to predict Titanic passenger survival, using Flask for the API and Docker for containerization. The model is trained on the famous Titanic dataset, saved as a pickle file, and deployed via a REST API.

The project includes:

* Training a machine learning model.
* Building and exposing a REST API using Flask to make predictions.
* Containerizing the Flask app using Docker.
* Storing the model in AWS S3 for persistence and deploying the application to AWS.

## Features
* Logistic regression model to predict survival.
* REST API to make predictions based on passenger data.
* Fully containerized using Docker for easy portability and deployment.
* AWS S3 for model persistence and AWS EC2 for cloud deployment.

```
├── model/
|   ├── app
│       ├── app.py              # Flask app to expose API
│       ├── titanic.py          # Model training script
│       └── titanic_model.pkl   # Pre-trained machine learning model
│   └── requirements.txt        # Python dependencies
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Test data (optional)
|── infra/
│   ├── ec2_role.tf            # Define the IAM Role for EC2 
│   └── main.tf                # Contains Terraform block and provider block
|   └── policies.tf            # Create policy for titanic project user
├── Dockerfile              # Dockerfile to containerize the app
├── docker-compose.yml      # Docker Compose file for multi-container setup
├── README.md               # Project readme
└── frontend/
    ├── index.html          # Frontend UI
    ├── style.css           # CSS styling for the frontend
    └── nginx.conf          # Nginx configuration for reverse proxy
    └── Dockerfile          # Dockerfile to containerize web app

