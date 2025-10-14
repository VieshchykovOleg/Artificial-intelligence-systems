import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC  # Використовуємо SVC з RBF ядром
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Кодування ознак
label_encoders = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoders.append(encoder)

# Розділення на X та y
X_final = X_encoded[:, :-1].astype(int)
y_final = X_encoded[:, -1].astype(int)

# Розбиття на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=5)

# --- НОВИЙ БЛОК: Створення та навчання моделі з гаусовим (RBF) ядром ---
print("Навчання моделі з гаусовим ядром (RBF)...")
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Прогноз
y_test_pred = classifier.predict(X_test)

# Оцінка моделі
accuracy = 100 * accuracy_score(y_test, y_test_pred)
precision = 100 * precision_score(y_test, y_test_pred, average='weighted')
recall = 100 * recall_score(y_test, y_test_pred, average='weighted')
f1 = 100 * f1_score(y_test, y_test_pred, average='weighted')

print("\n--- Результати оцінки моделі ---")
print(f"Accuracy: {round(accuracy, 2)}%")
print(f"Precision: {round(precision, 2)}%")
print(f"Recall: {round(recall, 2)}%")
print(f"F1 score: {round(f1, 2)}%")

# Прогноз для нової точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White',
              'Male', '0', '0', '40', 'United-States']

input_data_encoded = [-1] * len(input_data)
encoder_count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        encoder = label_encoders[encoder_count]
        input_data_encoded[i] = int(encoder.transform([item])[0])
        encoder_count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

predicted_class_encoded = classifier.predict(input_data_encoded)
predicted_class = label_encoders[-1].inverse_transform(predicted_class_encoded)

print(f"\n--- Прогноз для тестової точки ---")
print(f"Прогнозований клас доходу: {predicted_class[0]}")