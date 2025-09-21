import pickle
import numpy as np

MODEL_PATH = "Diabetes_modelH1.pkl"

def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(model):
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    values = []
    for feature in features:
        val = float(input(f"{feature}: "))
        values.append(val)

    patient_array = np.array(values).reshape(1, -1)
    
    prediction = model.predict(patient_array)[0]
    result = "Diabetes" if prediction == 1 else "No Diabetes"

    print(f"Input values: {values}")
    print(f"Prediction: {result}")

if __name__ == "__main__":
    model = load_model()
    predict(model)
