import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import pandas as pd

df = pd.read_csv('iris.csv')
# Q4
num_data = 1000  # data points per class
X = np.random.uniform(-5, 5, [2, num_data]).astype(np.float32)
y = 10 * np.exp(-X[0, :] ** 2) + X[1, :] ** 2 - 7 > 0
X = X.T
y = y.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.astype(np.int64)

y_test = y_test.astype(np.int64)

N = y_train.shape[0]
N_test = y_test.shape[0]

# make X and y torch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

model = nn.Sequential(
    nn.Linear(2, 50),
    nn.ReLU(),
    nn.Linear(50, 2),
)
# training loop
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

ls = []
for epoch in range(5000):
    a = model(X_train)
    # print(torch.Tensor.type(a))
    print(torch.Tensor.type(y_train))
    loss = loss_fn(a, y_train)
    loss.backward()  # computes the gradient
    optimizer.step()  # update params by -lr * gradient
    optimizer.zero_grad()  # zero out the gradient for next iteration
    ls.append(loss.item())

plt.figure()
plt.plot(ls)
plt.show()

y_hat = nn.functional.softmax(model(X_train), dim=1)
pred = torch.argmax(y_hat, dim=1)
accuracy = torch.sum((pred == y_train)) / N
print('Acc train: ', accuracy.item() * 100, '%')
print('Loss is :', ls[-1])
y_hat = nn.functional.softmax(model(X_test), dim=1)
pred = torch.argmax(y_hat, dim=1)
accuracy = torch.sum((pred == y_test)) / N_test
print('Acc test: ', accuracy.item() * 100, '%')
a = model(X_test)
loss = loss_fn(a, y_test)
print('Loss is :', loss.item())
