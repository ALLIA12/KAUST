import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import pandas as pd
import math 

df = pd.read_csv('./data/iris.csv')
feature_names = df.keys()
X = df.iloc[:, 0:4].to_numpy().astype(np.float32)
y = df.iloc[:, 4].to_numpy()
y[ y == 'Iris-setosa' ] = 0
y[ y == 'Iris-versicolor' ] = 1
y[ y == 'Iris-virginica' ] = 2
y = y.astype(np.int64)
N = y.shape[0]

# make X and y torch tensors
X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Split the training data into batches
batch_size = 15
batches = math.ceil(N / batch_size) 
X_batched = torch.tensor_split(X, [i*batch_size for i in range(1,batches)])
y_batched = torch.tensor_split(y, [i*batch_size for i in range(1,batches)])


model = nn.Linear(4, 3)

# training loop
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-3)  # add momentum?

ls = []
for epoch in range(5000):
    for b in range(batches):
        z = model(X_batched[b])
        loss = loss_fn(z, y_batched[b]) 
        loss.backward()            # computes the gradient
        optimizer.step()           # update params by -lr * gradient
        optimizer.zero_grad()      # zero out the gradient for next iteration
    ls.append(loss.item())

plt.figure()
plt.plot(ls)



y_hat = nn.functional.softmax(model(X), dim=1)
pred =  torch.argmax(y_hat, dim=1)
accuracy = torch.sum((pred == y)) / N 
print('accuracy: ', accuracy.item()*100, '%')

