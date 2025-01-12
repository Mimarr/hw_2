"""
В этом задании вам предстоит обучить модель для прогнозироввания стоимости недвижимости и для нее интерфейс в Streamlit.

Вам нужно:
Скачать данные и обучить модель прогнозирования стоимости недвижимости (модель может быть любой сложности,
даже линейная регрессия на двух признаках) - 2 балла +
Реализуйте код для получения предсказания обученной моделью - 1 балл +
Реализуйте интерфейс с помощью streamlit для введения значений признаков для прогнозирования - 5 баллов +
Реализуйте отображение результата прогнозирования в интерфейсе по нажатию кнопки - 2 балла +
"""

import os
import pickle


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Чтение датасета
file = pd.read_csv("realty_data.csv")
file.info()
file

def prepare_data():
    train = file[['price', 'total_square', 'floor', 'source', 'rooms', 'city']]  # , 'lat', 'lon'
    train = pd.get_dummies(train, drop_first=False)
    train['price'] = train['price'].astype(int)
    train['total_square'] = train['total_square'].astype(int)
    train['floor'] = train['floor'].astype(int)
    train = train.dropna()

    return train
    #print(train)

# Обучение модели
def model_train(train):
    X,y = train.drop('price', axis=1), train['price']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2025, test_size=0.1)

    lr = LinearRegression()
    lr.fit(X, y)

    with open('lr_fitted.pkl', 'wb') as file:
        pickle.dump(lr, file)

# Чтение модели
def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model


#model_train(prepare_data())
#read_model('D:\Python\pythonProject2_HW_8/lr_fitted.pkl')