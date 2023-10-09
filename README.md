# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent. Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

Use the standard libraries in python for finding linear regression.
Set variables for assigning dataset values.
Import linear regression from sklearn.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Obtain the graph.

# Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Gracia Ravi
RegisterNumber:  212222040047
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data1.txt", delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1], label="Admitted")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label ="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costfunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return j,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta =np.array([0,0,0])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
j,grad = costfunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  j = -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min()-1, X[:,0].max() +1
  y_min, y_max = X[:,1].min()-1, X[:,1].max() +1
  xx,yy =np.meshgrid(np.arange(x_min, x_max,0.1),np.arange(y_min,y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:,0],X[y== 1][:,1],label="Admitted")
  plt.scatter(X[y== 0][:,0],X[y ==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels =[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
# Output:
# Array Value of x
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/87de6675-3827-4dbd-950e-556a647786b6)


# Array Value of y
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/1b4aff91-e467-4f82-90ca-c1dba94b0a3f)


# Exam 1 - score graph
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/39297643-9752-4229-884e-b04c58ea6473)


# Sigmoid function graph
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/00827cce-da69-41b8-9da7-2c7c17a5dd42)


# X_train_grad value
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/0c6f9ee1-95da-440b-88f8-a898720cdbf3)


# Y_train_grad value
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/c968b72e-e823-4214-9b44-e9df3d1c037c)


# Print res.x
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/2095b300-8a1f-4f6e-b4a0-08bab5215e89)


# Decision boundary - graph for exam score
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/beef06a1-108e-46f6-825c-aa2be6eed00f)


# Probablity value
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/3ad8a497-e0f9-46e0-b73e-886877dfd6ff)


# Prediction value of mean
![image](https://github.com/gracia55/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/129026838/467be9e2-6de4-4c12-a3b1-11b138459235)


#  Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
