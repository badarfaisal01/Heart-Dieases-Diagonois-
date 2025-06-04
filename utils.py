import pandas as pd
import joblib

def preprocess_input(df):
    return df  # No heavy preprocessing here

def load_model():
    return joblib.load("models/heart_model.pkl")

def generate_report(data, prediction):
    with open("outputs/patient_report.txt", "w") as f:
        f.write("Patient Report\n")
        f.write("================\n")
        for key, val in data.items():
            f.write(f"{key}: {val}\n")
        result = "Positive (Heart Disease Detected)" if prediction == 1 else "Negative (No Heart Disease)"
        f.write(f"\nDiagnosis: {result}\n")
