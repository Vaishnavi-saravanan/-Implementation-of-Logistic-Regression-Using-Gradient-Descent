# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VAISHNAVI S
RegisterNumber:  212222230165
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```
## Output:
# Array Value of x:
![Screenshot 2023-10-03 150245](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/3a305aa9-310b-4624-92e5-0143cf7d5796)
# Array Value of y
![Screenshot 2023-10-03 150251](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/b37ad27a-822b-42db-bb7c-0ecc9be5b61f)
# Exam 1 - score graph
![Screenshot 2023-10-03 150302](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/65172e05-3688-4353-bae8-cd15774778c5)
# Sigmoid function graph
![Screenshot 2023-10-03 150311](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/63272e5c-485e-4b2e-a58c-a76daf39efd4)
# X_train_grad value
![Screenshot 2023-10-03 151000](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/121223bf-5f97-4f32-894d-93f19e88895c)

# Y_train_grad value
![Screenshot 2023-10-03 150324](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/a967ec72-9120-4b6d-a7fb-af299afb4494)

# Print res.x
![Screenshot 2023-10-03 150329](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/2a182451-72e9-4bd2-9756-d3ba667b8318)
# Decision boundary - graph for exam score
![Screenshot 2023-10-03 150336](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/a22cd28a-08dc-430c-9673-64494b8c08ea)
# Proability value
![Screenshot 2023-10-03 150342](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/e87f7ed2-3642-4965-b1be-fcdfd5905abb)
# Prediction value of mean
![Screenshot 2023-10-03 150347](https://github.com/Vaishnavi-saravanan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541897/337f4841-52e9-426f-a615-8e096824de57)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

