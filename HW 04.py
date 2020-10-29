from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv (r'C:\Users\andre\Documents\GitHub\MSE-307-HW-04\Data.csv')

t_630K = np.log(data['t (630K)'])
t_700K = np.log(data['t (700K)'])
y = np.log(np.log(1 / (1 - data['y'])))

lr_1 = LinearRegression().fit(t_630K.values.reshape(-1,1),y.values.reshape(-1,1))
lr_2 = LinearRegression().fit(t_700K.values.reshape(-1,1),y.values.reshape(-1,1))

x_1 = np.linspace(2.1,4.05,2)
x_2 = np.linspace(0.75,2.75,2)
y_1 = lr_1.coef_[0,0] * x_1 + lr_1.intercept_[0]
y_2 = lr_2.coef_[0,0] * x_2 + lr_2.intercept_[0]

print(lr_1.coef_[0,0])
print(lr_2.coef_[0,0])

fig_1 = plt.figure(dpi=300)
plt.scatter(t_630K, y, linewidth=1, label='630K: n=2.11')
plt.scatter(t_700K, y, linewidth=1, label='700K: n=2.04')
plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
    
plt.title('Glass to Crystal Transformation')
plt.xlabel('ln(Time)')
plt.ylabel(r'$\ln{\left[\ln{\left(\frac{1}{1 - Trans.}\right)}\right]}$')
plt.legend()

fig_1.savefig('Plot.png')