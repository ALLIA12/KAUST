import numpy as np
import matplotlib.pyplot as plt

# generate synthetic training data
num_data=100               # data points per class
N = 2 * num_data
x1=np.random.randn(num_data, 2) + 4.0
x0=np.random.randn(num_data, 2)
y1=np.ones(num_data)
y0=np.zeros(num_data)
X = np.concatenate((x1,x0))
X = np.concatenate((np.ones((N,1)), X), axis=1)
y = np.concatenate((y1,y0))

plt.scatter(X[:, 1], X[:, 2], c=y, s=20)
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(yhat, y):
  return np.sum(-y*np.log(yhat) - (1-y)*np.log(1-yhat))

lr = 1e-3
theta = np.random.randn(3)
ls = []
for i in range(30):
  # loss
  yhat = sigmoid(X @ theta)
  l = loss(yhat, y)
  ls.append(l)

  # gradient step: grad = dL/dtheta
  grad = X.T @ (yhat - y)
  theta -= lr * grad

plt.figure()
plt.plot(ls)
plt.show()

yhat = sigmoid(X @ theta)
prediction = (yhat >= 0.5)
accuracy = np.sum((prediction == y)) / N 
print('accuracy: ', accuracy*100, '%')

Nt = np.sum((y == 1) & (prediction == 1))
Nn = np.sum((y == 0) & (prediction == 0))
