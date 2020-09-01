import numpy as np #import modules
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = [[0],[3],[6],[9],[12], [15], [18]] #gets days in month for SA
y_train = [[0],[0],[1],[7],[16], [61], [116]] #num of infections

x_test = [[0],[3],[6],[9], [12], [15], [18]] #gets days in month for mexico
y_test = [[5],[6],[7],[7], [12], [43], [99]] #num of infections

regressor = LinearRegression() #declare regressor
regressor.fit(x_train, y_train)  #fits data
xx = np.linspace(0, 26, 100) #even spaced numbers
yy = regressor.predict(xx.reshape(xx.shape[0], 1)) #predict shape
plt.plot(xx,yy) #plots graph

quadratic_featurizer = PolynomialFeatures(degree=2) #projects data into a higher-dimensional space

x_train_quadratic = quadratic_featurizer.fit_transform(x_train) #transforms the data into quadratic data
x_test_quadratic = quadratic_featurizer.transform(x_test) #transforms the data into quadratic data

regressor_quadratic = LinearRegression() #declares the quadratic regressor
regressor_quadratic.fit(x_train_quadratic, y_train) # fits the data
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1)) #gets the shape 

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--') # plots the graph
plt.title('Infection rate in March') #title
plt.xlabel('Days in March') #xlabel
plt.ylabel('People Infected') #ylabel
plt.axis([0, 18, 0, 150]) # x and y axis
plt.grid(True) #adds a grid
plt.scatter(x_train, y_train)  #plots scatter points
plt.show() #displays graph
print (x_train)
print (x_train_quadratic)
print (x_test)
print (x_test_quadratic)
