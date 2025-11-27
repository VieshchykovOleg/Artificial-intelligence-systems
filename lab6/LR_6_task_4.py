import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

# Перетворюємо рядки дат у формат datetime
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Рахуємо різницю в часі та переводимо в хвилини
df['duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 60

# Очищення даних
df['price'] = df['price'].fillna(df['price'].mean())
df = df.dropna()

# Створення цільової змінної (Класифікація: Дешевий/Дорогий)
median_price = df['price'].median()
df['price_category'] = (df['price'] > median_price).astype(int)

# Кодування категоріальних ознак
le = LabelEncoder()
for col in ['train_type', 'train_class', 'fare', 'origin', 'destination']:
    df[col] = le.fit_transform(df[col])

# Вибір ознак (тепер 'duration' існує)
features = ['train_type', 'train_class', 'fare', 'duration']
X = df[features]
y = df['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, color='blue', bins=30)
plt.axvline(median_price, color='red', linestyle='--', label=f'Median Price ({median_price:.2f})')
plt.title('Розподіл цін на квитки')
plt.legend()
plt.show()