import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn import metrics 
import pickle 

MODEL_PATH = "Diabetes_modelH1.pkl"
def train(model_path=MODEL_PATH):
    data = pd.read_csv(r"C:\Users\alanf\Downloads\diabetes.csv")
    # print(data.head())

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age']
    target = 'Outcome'

    x = data[features]
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3) 
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2f}")

    # fig = plt.figure(figsize=(12,8))
    # tree.plot_tree(model, feature_names=features, class_names=['No Diabetes', 'Diabetes'], filled=True)
    # plt.show()
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

if __name__=="__main__":
    train()
