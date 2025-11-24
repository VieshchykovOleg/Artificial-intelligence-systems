import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
Y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# 1. Знаходження коефіцієнтів через матрицю Вандермонда
# Система рівнянь X_mat * A = Y
X_mat = np.vander(X, increasing=True)
A = np.linalg.solve(X_mat, Y)

# Створення функції полінома
P = np.poly1d(A[::-1]) # A[::-1] бо poly1d очікує від старшого степеня до молодшого

print("Коефіцієнти полінома:", A)
print(f"Значення P(0.2) = {P(0.2):.4f}")
print(f"Значення P(0.5) = {P(0.5):.4f}")

X_new = np.linspace(0.1, 0.7, 100)
Y_new = P(X_new)

plt.figure(figsize=(8, 6))
plt.plot(X, Y, 'ro', label='Задані точки')
plt.plot(X_new, Y_new, 'b-', label='Інтерполяційний поліном')
plt.plot(0.2, P(0.2), 'go', label=f'P(0.2)={P(0.2):.2f}')
plt.plot(0.5, P(0.5), 'go', label=f'P(0.5)={P(0.5):.2f}')

plt.title('Інтерполяція поліномом 4-го ступеня')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()