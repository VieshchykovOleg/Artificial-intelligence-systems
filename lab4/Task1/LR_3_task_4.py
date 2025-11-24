import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбивка (50% на 50% згідно завдання)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Навчання моделі
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Прогноз
ypred = regr.predict(Xtest)

# Вивід параметрів
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
print("R2 score =", r2_score(ytest, ypred))
print("Mean Absolute Error =", mean_absolute_error(ytest, ypred))
print("Mean Squared Error =", mean_squared_error(ytest, ypred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()