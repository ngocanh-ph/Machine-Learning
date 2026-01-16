import numpy as np
import matplotlib.pyplot as plt
X = np.array([[2, 4, 6, 8, 10]]).T
Y = np.array([4, 7, 10, 13, 16])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
A = np.dot(Xbar.T, Xbar)
B = np.dot(Xbar.T, Y)
w = np.dot(np.linalg.pinv(A), B)

beta_0 = w[0]
beta_1 = w[1]
Y_result = beta_0 + beta_1 * X
final_matrix = np.concatenate((X, Y_result), axis=1)

print(f"----- Result -----")
print(f"Coefficent beta_0: {beta_0:.2f}")
print(f"Coefficent beta_1: {beta_1:.2f}")
print("\nMa tran:\n", final_matrix)
x_predict = 7
y_predict = beta_0 + beta_1 * x_predict
print(f"Du doan: Hoc {x_predict} gio se duoc {y_predict:.2f} diem")

plt.figure(figsize = (8, 6))
plt.scatter(X, Y, color='blue', s=80, label ='Du lieu thuc te')
plt.plot (X, Y_result, color='red', linewidth=2, label=f'Duong hoi quy: y={y_predict:.1f}')
plt.scatter (x_predict, y_predict, color='green',label =f'Du doan 7h ({y_predict}đ)')
plt.title ("Bieu do quan he giua gio hoc va diem thi")
plt.xlabel("So gio hoc")
plt.ylabel ("Diem thi")
plt.grid(True)
plt.show()

