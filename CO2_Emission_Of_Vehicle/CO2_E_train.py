import matplotlib.pyplot as plt 
import pandas as pd
import pylab as pl 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import r2_score 
import pickle 

Model_path = "CO2_modelH1.pkl"
def train( model_path=Model_path):
   data = pd.read_csv("FuelConsumptionCo2.csv")
   print(data.describe())

   x = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
   y = data['CO2EMISSIONS']

   x_train, x_test, y_train, y_test = train_test_split(
      x,y, test_size=0.2, random_state=42
   )
   model = linear_model.LinearRegression()
   model.fit(x_train, y_train)

   print(f"the r2 score is: {r2_score(y_test, model.predict(x_test)):.2f}")

   sub_data = data[['CYLINDERS','CO2EMISSIONS']]
   plt.scatter(sub_data.CYLINDERS, sub_data.CO2EMISSIONS, color='blue')
   plt.xlabel("CYLINDER")
   plt.ylabel("CO2_EMISSION")
   plt.show()

   with open(model_path, "wb") as f:
      pickle.dump(model, f)
   print("model saved")
if __name__ == "__main__":
   train()

