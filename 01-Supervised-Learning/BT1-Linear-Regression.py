import numpy as np
import matplotlib.pyplot as plt
X = np.array([[147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 52, 54, 56, 58, 59, 60, 72, 63, 64, 66, 57, 68])
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
w0, w1 = w[0], w[1]
y1 = w1*155 + w0
y2 = w1*160 + w0
print('Input 155cm, true output 52kg, predicted output %.2fkg' % y1)
print('Input 160cm, true output 56kg, predicted output %.2fkg' % y2)
#Vẽ đồ thị 
x_line = np.linspace(X.min(), X.max(), 100)
y_line = w1 * x_line + w0
plt.scatter(X, y, color = 'red')
plt.plot(x_line, y_line, color= 'blue')                      
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Linear Regression: Height vs Weight")
plt.show()