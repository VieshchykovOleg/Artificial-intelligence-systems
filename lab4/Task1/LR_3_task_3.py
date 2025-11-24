import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 1. Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# 2. Поліноміальна регресія (ступінь 10)
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Тестова точка з завдання
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

print("\nLinear regression prediction:", linear_regressor.predict(datapoint))
print("Polynomial regression prediction:", poly_linear_model.predict(poly_datapoint))

# Метрики для лінійної моделі
y_test_pred = linear_regressor.predict(X_test)
print("\nLinear Regressor performance:")
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))