import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('SLR-Data.csv')
data.shape
print(data.head())
X = data.iloc[:,0].values
Y = data.iloc[:,1].values
mean_x = np.mean(X)
mean_y = np.mean(Y)
m = len(X)
numer = 0 
deom = 0 
for i in range(m):
    numer += (X[i] - mean_x ) * (Y[i] - mean_y)
    deom  += (X[i] - mean_x) ** 2
b1 = numer / deom
b0 = mean_y - (b1 * mean_x)
print(b1)
print(b0)
max_x = np.max(X)
min_x = np.min(X)
x = np.linspace(min_x , max_x)
y = b0 + b1 * x
plt.plot(x ,y , color = 'green' , label = 'RegressionLine')
plt.scatter(X ,Y,c='blue',label='Scatter')
plt.xlabel('No of Hours Spent During')
plt.ylabel('Risk Score on a scale of (0-100)')
plt.legend()
plt.show()
predict_x =int(input('Enter the no of Hours'))
predict_y =(4.58789860997547 * predict_x + 12.584627964022893)
plt.scatter(X ,Y)
plt.scatter(predict_x , predict_y)
plt.xlabel('No of hours spent')
plt.ylabel('Risk Score')
plt.scatter(X , Y , color='blue')
plt.plot(x ,y, color = 'green')
plt.show()
