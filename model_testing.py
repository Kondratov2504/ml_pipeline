import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

# Загрузка модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test, y_test = [], []

for file in os.listdir('test'):
    if file.startswith('prep_'):
        df = pd.read_csv(os.path.join('test', file))
        X_test.append(df[['day']])
        y_test.append(df['temperature'])

X_test = pd.concat(X_test)
y_test = pd.concat(y_test)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Тестирование завершено. Среднеквадратичная ошибка (MSE): {mse:.4f}")
