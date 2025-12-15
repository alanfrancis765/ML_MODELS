import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score 

data = load_iris()
dt = pd.DataFrame(data.data, columns=data.feature_names)
dt['target'] = data.target

for species, label in zip(iris.target_names, range(3)):
    subset = df[df["target"] == label]
    plt.scatter(
        subset["sepal length (cm)"],
        subset["sepal width (cm)"],
        label=species
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset Scatter Plot")
plt.legend()
plt.show()

x = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

param_grid = {
    "model__n_neighbors": [4, 3, 10, 5, 15, 20] 
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv = 3,
    scoring="accuracy"
)
grid.fit(x_train, y_train)
print(f"Best CV Accuracy: {grid.best_score_:.2f}")
print("Test Accuracy:", grid.score(x_test, y_test))
