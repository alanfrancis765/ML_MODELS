import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits

dt = load_digits()
data = pd.DataFrame(dt.data , columns = dt.feature_names)
data['target'] = dt.target

#PCA for the group wise visualization 
df = data.copy()
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plot = plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=y,
    cmap="tab10",
    alpha=0.7
)
plt.colorbar(plot, label="Digit Label")
plt.show()

#model training 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import confusion_matrix

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter= 1000, 
                      random_state = 42)

model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.4f}")
print(f"classification Report:\n{classification_report(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - MLP Digits Classifier")
plt.show()

