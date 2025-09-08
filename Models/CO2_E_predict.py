import pickle 
import numpy as np

Model_path = "CO2_model.pkl"
def predict(feature, model_path=Model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        return model.predict(np.array([feature]))
    
if __name__ == "__main__":
    y  = input("Enter ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB: ")
    v = list(map(float, y.split()))
    result = predict(v)
    print(f"CO2EMISSION for the {v}: {result[0]:.2f}")