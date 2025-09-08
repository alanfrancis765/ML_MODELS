import numpy as np
import pandas as pd 
import pickle 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 

Model_path = "CO2_model.pkl"
def train(model_path=Model_path):
   data = pd.DataFrame ({
    "ENGINESIZE": [2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7],
    "CYLINDERS": [4, 4, 4, 6, 6, 6, 6, 6, 6],
    "FUELCONSUMPTION_COMB": [8.5, 9.6, 5.9, 11.1, 10.6, 10.0, 10.1, 11.1, 11.6],
    "CO2EMISSIONS": [196, 221, 136, 255, 244, 230, 232, 255, 267]
 })

   X_train = data[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]]
   Y_train = data["CO2EMISSIONS"]

   model = LinearRegression()
   model.fit(X_train, Y_train)

   r2_score_val = r2_score(Y_train, model.predict(X_train))
   print(f"R² score: {r2_score_val:.2f}")

   with open("CO2_model.pkl", "wb") as f:
    pickle.dump(model, f)


if __name__ == "__main__":
      train()