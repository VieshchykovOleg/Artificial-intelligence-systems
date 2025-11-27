import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Дані
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Кодування
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('Play', axis=1)
y = df['Play']

# Навчання моделі
model = CategoricalNB()
model.fit(X, y)

# Варіант 8: Sunny, High, Weak
# Потрібно закодувати ці значення так само, як це зробив LabelEncoder
# Припустимо: Sunny=2, High=0, Weak=1 (залежить від алфавітного порядку)
test_sample = [[2, 0, 1]] 

prob = model.predict_proba(test_sample)
print(f"Ймовірність No: {prob[0][0]:.4f}")
print(f"Ймовірність Yes: {prob[0][1]:.4f}")
print(f"Прогноз: {'Yes' if prob[0][1] > 0.5 else 'No'}")