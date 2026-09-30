# Diabetes Prediction using KNN

A Machine Learning project that predicts whether a person is likely to have diabetes using the Pima Indians Diabetes Dataset and the K-Nearest Neighbors (KNN) classification algorithm.

## Project Overview

This project uses patient health-related information to build a machine learning classification model for diabetes prediction.

The dataset contains 768 records with 8 input features and 1 target variable.

The project covers:

- Data loading and exploration
- Data preprocessing
- Feature scaling
- KNN classification
- Hyperparameter tuning
- Model evaluation
- Model serialization using Pickle

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains the following features:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Target variable |

`Outcome` is the target variable:

- `0` → No diabetes
- `1` → Diabetes

## Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification.

For a new patient, the model compares the patient's feature values with nearby training samples and predicts the class based on the nearest neighbors.

## Preprocessing

Feature scaling is performed using:

```python
StandardScaler()
