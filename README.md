# Data-Sains
Tubes Data Sains Link: https://data-sains-test.streamlit.app/

# Obesity Level Prediction

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

This project is a Data Science Final Assignment (Tubes) aimed at predicting an individual's obesity level based on their lifestyle, eating habits, and physical attributes using Machine Learning.

The application is built using **Streamlit** for the user interface and **Random Forest** as the prediction model.

## 📋 Features

The application accepts the following user inputs:
* **Physical Attributes:** Age, Gender, Height, Weight.
* **Health History:** Family history with overweight.
* **Eating Habits:**
    * High caloric food consumption (FAVC).
    * Frequency of vegetable consumption (FCVC).
    * Number of main meals per day (NCP).
    * Consumption of food between meals (CAEC).
    * Alcohol consumption (CALC).
    * Daily water consumption (CH2O).
* **Lifestyle:**
    * Smoking habit (SMOKE).
    * Caloric beverages consumption (SCC).
    * Physical activity frequency (FAF).
    * Time using technology devices (TUE).
    * Main transportation mode (MTRANS).

**Prediction Output:**
The model classifies the user's condition into one of the following categories:
* Insufficient Weight
* Normal Weight
* Overweight Level I & II
* Obesity Type I, II, & III

## 📂 File Structure

* `app.py`: The main Streamlit application file handling the interface and prediction logic.
* `random_forest_obesity_model.joblib`: The trained Machine Learning model (Random Forest).
* `preprocessing_objects.joblib`: Scaler objects for normalizing input data to match the training data.
* `requirements.txt`: List of required Python libraries.
* `Data_Science_Obesity_Risk.ipynb`: Jupyter Notebook containing the Exploratory Data Analysis (EDA) and model training process.


