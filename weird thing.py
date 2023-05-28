import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import pandas as pd
import time

device = torch.device('cuda:0')
# df = pd.read_csv('iris.csv')
# Q4
num_data = 1000  # data points per class
X = np.random.uniform(-5, 5, [2, num_data]).astype(np.float32)
y = (np.floor(X[0, :] % 2) == 0) & (np.floor(X[1, :] % 2) == 0) > 0
X = X % 2

X = X.T
y = y.T
np.random.seed(42)
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

X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

numClasses = 2
layer1Features = X.T.shape[0]
layer2Features = 32
model = nn.Sequential(
    nn.Linear(layer1Features, layer2Features),
    nn.ReLU(),
    nn.Linear(layer2Features, numClasses),
).to(device)
# training loop
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1)
start_time = time.time()

ls = []
for epoch in range(500000):
    a = model(X_train)
    # print(torch.Tensor.type(a))
    loss = loss_fn(a, y_train)
    loss.backward()  # computes the gradient
    optimizer.step()  # update params by -lr * gradient
    optimizer.zero_grad()  # zero out the gradient for next iteration
    ls.append(loss.item())

end_time = time.time()
elapsed_time = end_time - start_time

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

print("Time taken:", elapsed_time, "seconds")
