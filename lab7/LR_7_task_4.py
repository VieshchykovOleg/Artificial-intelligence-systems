import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf

input_file = 'company_symbol_mapping.json'

with open(input_file, 'r') as f:
    company_symbols_map = json.load(f)

names, symbols = np.array(list(company_symbols_map.items())).T

start_date = "2003-07-03"
end_date = "2007-05-04"

quotes = []
valid_names = []

print("Завантаження котирувань...")
for symbol, name in zip(symbols, names):
    try:
        data = yf.download(symbol, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)

        if not data.empty:
            quotes.append(data)
            valid_names.append(name)
        else:
            print(f"Немає даних для {name} ({symbol})")
    except Exception as e:
        print(f"Помилка для {name}: {e}")
names = np.array(valid_names)

if len(quotes) == 0:
    print("Помилка: Не вдалося завантажити жодних даних. Перевірте інтернет або символи.")
    exit()

quotes_aligned = pd.concat(quotes, axis=1, keys=names)

# Витягуємо Open/Close для всіх компаній
opening_quotes = quotes_aligned.loc[:, (slice(None), 'Open')].values
closing_quotes = quotes_aligned.loc[:, (slice(None), 'Close')].values

quotes_diff = closing_quotes - opening_quotes

# Нормалізація та очищення
X = quotes_diff.copy()
X = np.nan_to_num(X)   # замінює NaN та inf
X = X[~np.all(X == 0, axis=1)]
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Кластеризація
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print("\nРезультати кластеризації:")
for i in range(num_labels + 1):
    cluster_members = names[labels == i]
    print(f"Cluster {i + 1} ==> {', '.join(cluster_members)}")