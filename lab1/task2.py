def activation(x):
    return 1 if x >= 0 else 0


# Персептрон для функції OR
def or_perceptron(x1, x2):
    """
    OR персептрон з вагами W1=1, W2=1, W0=-0.5
    Розділяюча лінія: x1 + x2 - 0.5 = 0
    """
    W1, W2, W0 = 1, 1, -0.5
    net = W1 * x1 + W2 * x2 + W0
    return activation(net)


# Персептрон для функції AND
def and_perceptron(x1, x2):
    """
    AND персептрон з вагами W1=1, W2=1, W0=-1.5
    Розділяюча лінія: x1 + x2 - 1.5 = 0
    """
    W1, W2, W0 = 1, 1, -1.5
    net = W1 * x1 + W2 * x2 + W0
    return activation(net)


# Реалізація XOR через двошаровий персептрон
def xor_network(x1, x2):
    """
    XOR = OR(x1, x2) AND NOT(AND(x1, x2))

    Перший шар:
    - y1 = OR(x1, x2)
    - y2 = AND(x1, x2)

    Другий шар:
    - output = y1 AND NOT(y2)
    """
    # Перший шар
    y1 = or_perceptron(x1, x2)  # OR
    y2 = and_perceptron(x1, x2)  # AND

    # Другий шар: y1 AND NOT(y2)
    # Це еквівалентно: y1 - y2 >= 0.5
    W1_out, W2_out, W0_out = 1, -1, -0.5
    net_out = W1_out * y1 + W2_out * y2 + W0_out
    output = activation(net_out)

    return output, y1, y2


# Тестування всіх можливих входів
print("=" * 60)
print("ТЕСТУВАННЯ НЕЙРОННОЇ РЕАЛІЗАЦІЇ XOR")
print("=" * 60)

print("\n1. Перевірка базових функцій OR та AND:")
print("-" * 60)
print("x1 | x2 | OR(x1,x2) | AND(x1,x2)")
print("-" * 60)

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x1, x2 in test_inputs:
    or_result = or_perceptron(x1, x2)
    and_result = and_perceptron(x1, x2)
    print(f" {x1} |  {x2} |     {or_result}     |      {and_result}")

print("\n2. Повна таблиця істинності для XOR:")
print("-" * 60)
print("x1 | x2 | y1=OR | y2=AND | XOR(x1,x2)")
print("-" * 60)

for x1, x2 in test_inputs:
    xor_result, y1, y2 = xor_network(x1, x2)
    print(f" {x1} |  {x2} |   {y1}   |   {y2}    |     {xor_result}")

print("\n3. Перевірка правильності:")
print("-" * 60)
expected_xor = [0, 1, 1, 0]
all_correct = True

for i, (x1, x2) in enumerate(test_inputs):
    xor_result, _, _ = xor_network(x1, x2)
    is_correct = (xor_result == expected_xor[i])
    all_correct = all_correct and is_correct
    status = "✓" if is_correct else "✗"
    print(f"XOR({x1}, {x2}) = {xor_result}, очікувалось {expected_xor[i]} {status}")

print("-" * 60)
if all_correct:
    print("✓ ВСІ ТЕСТИ ПРОЙДЕНО УСПІШНО!")
else:
    print("✗ Виявлено помилки в реалізації")

print("\n4. Архітектура двошарового персептрона:")
print("-" * 60)
print("""
Вхідний шар:        x1, x2

Прихований шар:     y1 = OR(x1, x2)  [W1=1, W2=1, W0=-0.5]
                    y2 = AND(x1, x2) [W1=1, W2=1, W0=-1.5]

Вихідний шар:       XOR = f(y1 - y2 - 0.5)
                    [W1=1, W2=-1, W0=-0.5]
""")
print("=" * 60)