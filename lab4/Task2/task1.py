import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X_data = np.array([[6], [7], [8], [9], [10], [12]])
Y_data = np.array([2, 3, 3, 4, 6, 5])

# Створення та навчання моделі
model = LinearRegression()
model.fit(X_data, Y_data)

# Отримання коефіцієнтів
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print(f"Коефіцієнт нахилу (beta_1): {beta_1}")
print(f"Вільний член (beta_0): {beta_0}")

# Побудова графіка
X_fit = np.linspace(X_data.min(), X_data.max(), 100).reshape(-1, 1)
Y_pred = model.predict(X_fit)

plt.figure(figsize=(8, 6))
plt.scatter(X_data, Y_data, color='red', s=100, label='Експериментальні точки (Вар. 8)')
plt.plot(X_fit, Y_pred, color='blue', linewidth=2, label=f'Апроксимація: y = {beta_0:.2f} + {beta_1:.2f}x')
plt.title('Лінійна регресія для Варіанту 8')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()