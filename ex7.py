import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import math 
from sklearn import datasets, model_selection, preprocessing
from torchinfo import summary
import pandas as pd


# read MNIST training data
df = pd.read_csv('./data/mnist_train.csv')
X = df.iloc[:, 1:].to_numpy() / 255.0       # values are scaled to be between 0 and 1
y = df.iloc[:, 0].to_numpy()                # labels of images
N, D = X.shape

# make X and y torch tensors and batch them
X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y)
batch_size = 64
batches = math.ceil(N / batch_size) 
X_batched = torch.tensor_split(X, [i*batch_size for i in range(1,batches)])
y_batched = torch.tensor_split(y, [i*batch_size for i in range(1,batches)])

# plot the first dozen images from the data set
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1, xticks=[], yticks=[])
    image = X[i, :].reshape((28,28))
    plt.imshow(image, cmap='gray')

plt.show()


model = nn.Sequential(
    nn.Linear(D, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10) 
) 

print(model)
summary(model, input_size=(batch_size, D))


# test a few images to see if predicted labels are correct
for k in [1, 5, 10, 200]:
    y_hat = nn.functional.softmax(model(X[k]), dim=0)
    pred =  torch.argmax(y_hat, dim=0)
    print(pred.item(), y[k].item())


# training loop
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
epochs = 2
ls = []
for epoch in range(epochs):
    for b in range(batches):
        z = model(X_batched[b])
        loss = loss_fn(z, y_batched[b]) 
        loss.backward()            # computes the gradient
        optimizer.step()           # update params by -lr * gradient
        optimizer.zero_grad()      # zero out the gradient for next iteration
        if b % 100 == 0:
            print(loss.item())



# test a few images to see if predicted labels are correct
for k in [1, 5, 10, 200]:
    y_hat = nn.functional.softmax(model(X[k]), dim=0)
    pred =  torch.argmax(y_hat, dim=0)
    print(pred.item(), y[k].item())
