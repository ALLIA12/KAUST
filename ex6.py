import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import math 
from sklearn import datasets, model_selection, preprocessing
from torchinfo import summary

# datasets.fetch_california_housing?
features, targets = datasets.fetch_california_housing(return_X_y=True, as_frame=True)
# features.info() 
# features.describe()
# targets.describe()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    features, targets, test_size=0.1)
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# make X and y torch tensors
X = torch.from_numpy(X_train.to_numpy(dtype='float32'))
y = torch.from_numpy(y_train.to_numpy(dtype='float32'))
# X = torch.from_numpy(X_train)
# y = torch.from_numpy(y_train)


# Split the training data into batches
N, D = X.shape
batch_size = 100
batches = math.ceil(N / batch_size) 
X_batched = torch.tensor_split(X, [i*batch_size for i in range(1,batches)])
y_batched = torch.tensor_split(y, [i*batch_size for i in range(1,batches)])

H = 32
model = nn.Sequential(
    nn.Linear(D, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    nn.Linear(H, H),
    nn.ReLU(),
    nn.Linear(H, 1)        
) 

print(model)
summary(model, input_size=(batch_size, D))

# training loop
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=5e-6)
epochs = 2000
ls = []
for epoch in range(epochs):
    for b in range(batches):
        z = model(X_batched[b])
        loss = loss_fn(z.squeeze(), y_batched[b])
        loss.backward()            # computes the gradient
        optimizer.step()           # update params by -lr * gradient
        optimizer.zero_grad()      # zero out the gradient for next iteration
    ls.append(loss.item())
    if epoch % 10 == 0:
        print(loss.item())

plt.figure()
plt.plot(ls)

def rms(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))

yhat = model(X).squeeze()
e = rms(yhat.detach().numpy(), y.detach().numpy())