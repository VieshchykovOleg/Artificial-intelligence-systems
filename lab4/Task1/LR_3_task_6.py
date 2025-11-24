import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

m = 100
X = np.linspace(-3, 3, m)
noise = np.random.uniform(-0.5, 0.5, m)

y = 2 * np.sin(X) + noise

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []

    # Тренуємо модель на зростаючому наборі даних
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    plt.ylabel("RMSE")
    plt.xlabel("Розмір навчального набору")
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(10, 6))
linear_reg = LinearRegression()
plot_learning_curves(linear_reg, X, y)
plt.title("Learning Curves (Linear Regression)")
plt.show()

polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

plt.figure(figsize=(10, 6))
plot_learning_curves(polynomial_regression_10, X, y)
plt.axis([0, 80, 0, 1.5])
plt.title("Learning Curves (Polynomial Degree 10)")
plt.show()

polynomial_regression_opt = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

plt.figure(figsize=(10, 6))
plot_learning_curves(polynomial_regression_opt, X, y)
plt.axis([0, 80, 0, 1.5])
plt.title("Learning Curves (Polynomial Degree 3)")
plt.show()