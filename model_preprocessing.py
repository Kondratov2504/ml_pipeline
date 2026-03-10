import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(folder):
    scaler = StandardScaler()
    for file in os.listdir(folder):
        if file.endswith('.csv') and not file.startswith('prep_'):
            filepath = os.path.join(folder, file)
            df = pd.read_csv(filepath)
            # Масштабируем температуру
            df[['temperature']] = scaler.fit_transform(df[['temperature']])
            df.to_csv(os.path.join(folder, f'prep_{file}'), index=False)

preprocess_data('train')
preprocess_data('test')
