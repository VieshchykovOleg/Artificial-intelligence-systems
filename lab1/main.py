import matplotlib.pyplot as plt
import numpy as np


# Функції AND, OR, XOR
def and_gate(x1, x2):
    """AND функція"""
    return 1 if (x1 == 1 and x2 == 1) else 0


def or_gate(x1, x2):
    """OR функція"""
    return 1 if (x1 == 1 or x2 == 1) else 0


def xor_gate(x1, x2):
    """XOR функція через AND і OR: XOR = OR AND (NOT AND)"""
    or_result = or_gate(x1, x2)
    and_result = and_gate(x1, x2)
    # XOR = OR але не AND
    return 1 if (or_result == 1 and and_result == 0) else 0


# Тестові дані
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("Таблиця істинності:")
print("x1 | x2 | AND | OR | XOR")
print("-" * 25)

# Результати для графіків
x1_vals, x2_vals = [], []
and_results, or_results, xor_results = [], [], []

for x1, x2 in inputs:
    and_res = and_gate(x1, x2)
    or_res = or_gate(x1, x2)
    xor_res = xor_gate(x1, x2)

    print(f" {x1} |  {x2} |  {and_res}  |  {or_res} |  {xor_res}")

    x1_vals.append(x1)
    x2_vals.append(x2)
    and_results.append(and_res)
    or_results.append(or_res)
    xor_results.append(xor_res)

# Створення графіків
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# График AND
axes[0].scatter(range(4), and_results, c=['red' if r == 0 else 'green' for r in and_results], s=100)
axes[0].set_title('AND функція')
axes[0].set_xlabel('Комбінація входів')
axes[0].set_ylabel('Вихід')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
axes[0].set_ylim(-0.5, 1.5)
axes[0].grid(True, alpha=0.3)

# График OR
axes[1].scatter(range(4), or_results, c=['red' if r == 0 else 'green' for r in or_results], s=100)
axes[1].set_title('OR функція')
axes[1].set_xlabel('Комбінація входів')
axes[1].set_ylabel('Вихід')
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
axes[1].set_ylim(-0.5, 1.5)
axes[1].grid(True, alpha=0.3)

# График XOR
axes[2].scatter(range(4), xor_results, c=['red' if r == 0 else 'green' for r in xor_results], s=100)
axes[2].set_title('XOR функція (через AND/OR)')
axes[2].set_xlabel('Комбінація входів')
axes[2].set_ylabel('Вихід')
axes[2].set_xticks(range(4))
axes[2].set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
axes[2].set_ylim(-0.5, 1.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Демонстрація алгоритму XOR
print("\nДемонстрація алгоритму XOR:")
print("XOR(x1, x2) = OR(x1, x2) AND NOT AND(x1, x2)")
print()

for x1, x2 in inputs:
    or_val = or_gate(x1, x2)
    and_val = and_gate(x1, x2)
    xor_val = xor_gate(x1, x2)
    print(f"XOR({x1},{x2}) = OR({x1},{x2})={or_val} AND NOT AND({x1},{x2})={1 - and_val} = {xor_val}")