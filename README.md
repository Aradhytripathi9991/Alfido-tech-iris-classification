# Alfido-tech-iris-classification


# 🌸 Iris Species Classification & Streamlit Web App

A machine learning application that performs Exploratory Data Analysis (EDA), trains and evaluates multiple classification algorithms (k-NN, Logistic Regression, Decision Tree), and serves interactive predictions via a Streamlit web interface.

---

## 📌 Project Overview

This project provides an end-to-end machine learning pipeline for classifying iris flowers into three species (*setosa*, *versicolor*, and *virginica*) based on four physical measurements:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 🚀 Features

- **Exploratory Data Analysis (EDA):** Interactive feature distributions, correlation heatmaps, and class separability pairplots.
- **Multi-Model Comparison:** Train and evaluate k-Nearest Neighbors, Logistic Regression, and Decision Tree models with customizable hyperparameters.
- **Evaluation Metrics:** Detailed accuracy scores, confusion matrices, and precision/recall classification reports.
- **Model Persistence:** Automatic saving of the top-performing model (`iris_best_model.pkl`) and scaler (`iris_scaler.pkl`) using `joblib`.
- **Live Inference Engine:** Real-time species prediction with class probability visualization for user-inputted sample features.

---

## 📁 Repository Structure

```text
.
├── app.py                 # Main Streamlit web application
├── iris.ipynb             # Jupyter Notebook with EDA, training, & experimentation
├── iris.csv               # Iris dataset (if local dataset file is used)
├── iris_best_model.pkl    # Serialized best-performing ML model
├── iris_scaler.pkl        # Serialized feature scaler (StandardScaler)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
