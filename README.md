# Disease_Prediction_System

This repository contains a **Disease Prediction System** that uses machine learning models to predict the likelihood of different diseases. The system is implemented in Python and deployed as a web app using Streamlit.

## Overview

The application predicts the presence of diseases like diabetes, heart disease, and Parkinson's disease based on user inputs. Each disease model is trained on specific features and saved using Pickle. The app provides a simple UI where users can input relevant health information, and the system will output a prediction based on the trained models.

## Features

- **Multiple Disease Prediction**: Currently supports Diabetes, Heart Disease, and Parkinson's Disease prediction.
- **User-Friendly Interface**: Streamlit-based UI with side navigation to select the disease prediction type.
- **Expandable**: Additional diseases and models can be added.

## Machine Learning Models

The system uses different machine learning models for each disease prediction. Each model is trained on a specific dataset and evaluated for accuracy.

### 1. Heart Disease Prediction

- **Dataset**: Heart Disease dataset containing health metrics like blood pressure, cholesterol levels, etc.
- **Model**: Logistic Regression
- **Preprocessing**: Checked for missing values, visualized data distributions, and split into training and test sets.
- **Evaluation**: Achieved high accuracy on both training and test sets.
- **Notebook**: [Heart_Disease_Prediction.ipynb](Heart_Disease_Prediction.ipynb)

### 2. Diabetes Prediction

- **Dataset**: PIMA Diabetes dataset with features like glucose levels, blood pressure, and BMI.
- **Model**: Support Vector Machine (SVM) with a linear kernel
- **Preprocessing**: Standardized the data and split it into training and test sets.
- **Evaluation**: The SVM model achieved good accuracy on the test set.
- **Notebook**: [Diabetes_Prediction.ipynb](Diabetes_Prediction.ipynb)

### 3. Parkinson’s Disease Detection

- **Dataset**: Parkinson's dataset containing voice measurements to detect Parkinson's disease.
- **Model**: Support Vector Machine (SVM) with a linear kernel
- **Preprocessing**: Standardized numeric data after handling categorical data and split it into training and test sets.
- **Evaluation**: High accuracy achieved on training and test sets, indicating strong model performance.
- **Notebook**: [Parkinson's_Disease_Detection.ipynb](Parkinson's_Disease_Detection.ipynb)

## Technologies Used

- **Python**: Core language for building and training the models.
- **Streamlit**: Web framework for deploying the app.
- **Pickle**: Used for saving and loading trained models.
- **Pandas**: For data manipulation.
- **Machine Learning Models**: Logistic Regression, Support Vector Machine (SVM).

  

