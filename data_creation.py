import os
import numpy as np
import pandas as pd

# Создание папок, если их нет
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

def generate_data(filename, size, noise_scale):
    days = np.arange(size)
    # Базовая температура 15 градусов + тренд + шум
    temperature = 15 + 0.1 * days + np.random.normal(0, noise_scale, size)
    df = pd.DataFrame({'day': days, 'temperature': temperature})
    df.to_csv(filename, index=False)

# Тренировочные данные (с разным уровнем шума)
generate_data('train/data_1.csv', 100, 2.0)
generate_data('train/data_2.csv', 150, 4.0) 
# Тестовые данные
generate_data('test/data_test.csv', 50, 2.0)
