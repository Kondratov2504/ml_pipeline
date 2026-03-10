import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X_train, y_train = [], []

for file in os.listdir('train'):
    if file.startswith('prep_'):
        df = pd.read_csv(os.path.join('train', file))
        X_train.append(df[['day']])
        y_train.append(df['temperature'])

X_train = pd.concat(X_train)
y_train = pd.concat(y_train)

model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение обученной модели
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
