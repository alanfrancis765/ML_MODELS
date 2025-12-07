from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as f 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dt = load_wine()
data = pd.DataFrame(dt.data, columns = dt.feature_names)
data['target'] = dt.target 

x = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 

class Model(nn.Module):
    def __init__(self, inputs= 13, h1 = 9, h2 = 9, h3 = 9, classes = 3):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(inputs, h1)
        self.fc2 = nn.Linear(h1, h2 )
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, classes)
    def forward(self , x):

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.out(x)
        return x


torch.manual_seed(42)
model = Model()

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

scaled = StandardScaler()
x_train_scaled  = scaled.fit_transform(x_train)
x_test_scaled = scaled.transform(x_test)

x_train = torch.FloatTensor(x_train_scaled)
x_test  = torch.FloatTensor(x_test_scaled)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion  = nn.CrossEntropyLoss() #how close we are to the Y
optimizer = torch.optim.Adam(model.parameters(), lr= 0.01) #adjust the weight 
# parameters: the weight and the bias that we can change 

epoch = 100 # One full pass through the entire training dataset by the model.
losses = []

for i in range(epoch):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item()) #.item() converts tensor → Python float

    if i % 10 == 0:
        print(f"Epoch {i} | loss: {loss: 4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epoch), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training loss Curve")

    
