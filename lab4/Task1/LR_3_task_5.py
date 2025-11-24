import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))

# Поліноміальні ознаки (ступінь 2 - спробуйте також 3 або 5 для синусоїди)
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Навчання лінійної регресії на поліноміальних даних
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Вивід коефіцієнтів
print("Intercept (вільний член):", lin_reg.intercept_)
print("Coefficients (коефіцієнти при ступенях):", lin_reg.coef_)

# Побудова графіка
# ВИПРАВЛЕННЯ 2: Змінив діапазон X_new на (-3, 3), щоб він відповідав вашим даним (у вас було -5, 1)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.", label="Дані (Варіант 8)")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Прогноз")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title("Поліноміальна регресія для y = 2*sin(X)")
plt.show()