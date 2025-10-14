import matplotlib
matplotlib.use('TkAgg')
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Завантаження даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# --- Аналіз даних ---
print("Форма датасету:", dataset.shape)
print("\nСтатистичне зведення:")
print(dataset.describe())
print("\nРозподіл за класами:")
print(dataset.groupby('class').size())

# --- Візуалізація ---
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.suptitle("Діаграма розмаху")
plt.show()

dataset.hist()
plt.suptitle("Гістограма розподілу")
plt.show()

scatter_matrix(dataset)
plt.suptitle("Матриця діаграм розсіювання")
plt.show()

# --- Розділення даних ---
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# --- Побудова моделей ---
models = []
# LogisticRegression через OneVsRestClassifier, щоб уникнути попереджень
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='lbfgs'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# --- Крос-валідація ---
results = []
names_list = []
print('\n--- Результати крос-валідації ---')
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_list.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# --- Порівняння алгоритмів ---
plt.boxplot(results, tick_labels=names_list)
plt.title('Algorithm Comparison')
plt.show()

# --- Оцінка SVM на контрольній вибірці ---
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print('\n--- Оцінка SVM на контрольній вибірці ---')
print("Точність:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("Звіт про класифікацію:\n", classification_report(Y_validation, predictions))

# --- Прогноз для нових даних ---
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)
print(f"\n--- Прогноз для квітки з параметрами {X_new[0]} ---")
print(f"Спрогнозований клас: {prediction[0]}")
