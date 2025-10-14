import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Крок 1: Підготовка даних (з Завдання 2.1)
print("--- Підготовка даних про доходи ---")

# Переконайтеся, що файл 'income_data.txt' знаходиться в тій самій папці
try:
    input_file = 'income_data.txt'
    X_list = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            # Пропускаємо порожні рядки, які можуть виникнути в кінці файлу
            if not line.strip():
                continue
            if '?' in line:
                continue
            data = line.strip().split(', ')
            X_list.append(data)
    X_np = np.array(X_list)
except FileNotFoundError:
    print(f"Помилка: Файл {input_file} не знайдено. Переконайтеся, що він у тій самій директорії.")
    exit()
except Exception as e:
    print(f"Сталася помилка при читанні файлу: {e}")
    exit()

# Кодування категоріальних ознак
label_encoders = []
X_encoded = np.empty(X_np.shape, dtype=object) # Використовуємо dtype=object для змішаних типів
for i, item in enumerate(X_np[0]):
    # Перевіряємо, чи всі значення у стовпці є числами
    try:
        X_encoded[:, i] = X_np[:, i].astype(float)
    except ValueError:
        # Якщо ні - це категоріальна ознака, кодуємо її
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X_np[:, i])
        label_encoders.append(encoder)

X_final = X_encoded[:, :-1].astype(float)
y_final = X_encoded[:, -1].astype(int)

# Розділення даних
X_train, X_validation, Y_train, Y_validation = train_test_split(X_final, y_final, test_size=0.20, random_state=1)
print("Підготовка даних завершена.")

# Крок 2: Порівняння моделей
print("\n--- Порівняння ефективності 6 алгоритмів ---")
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', LinearSVC(max_iter=2000, dual=False)))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Візуалізація результатів
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Порівняння алгоритмів для даних про доходи')
pyplot.show()

# Крок 3: Оцінка найкращої моделі
# Обираємо LDA, оскільки він зазвичай показує добрий результат на цих даних і є швидким
print("\n--- Детальна оцінка моделі LDA ---")
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("Точність:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))

# Знаходимо енкодер для цільової змінної (останній з текстових стовпців)
income_encoder = None
temp_encoders = []
for i, item in enumerate(X_np[0]):
    try:
        X_np[:, i].astype(float)
    except ValueError:
        le = preprocessing.LabelEncoder()
        le.fit(X_np[:, i])
        if i == X_np.shape[1] - 1:
            income_encoder = le

if income_encoder:
    class_names = income_encoder.classes_
    print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions, target_names=class_names))
else:
    print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions))