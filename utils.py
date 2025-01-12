import os

import pandas as pd
import streamlit as st

from main import prepare_data, model_train, read_model

st.set_page_config(
    page_title="Real Estate App",
)

#'price', 'total_square', 'floor', 'source', 'rooms', 'city'
model_path = 'lr_fitted.pkl'
total_square = st.sidebar.number_input("Площадь квартиры", 32, 500, 55)
floor = st.sidebar.number_input("Этаж", 1, 25, 1)
rooms = st.sidebar.slider("Количество комнат в квартире", 1, 10, 1, 1) #5, 6, 7, 8, 9, 10)
source = st.sidebar.selectbox("Источник информации (сайт)",
    ("Домклик", "Новострой-М", "ЦИАН", "Яндекс.Недвижимость"),
    index=1
)
city = st.sidebar.selectbox(
    "Место расположения",
    ("Балашиха", "Видное", "Дзержинский", "Долгопрудный", "Ивантеевка",
     "Королёв", "Котельники", "Красногорск", "Лобня", "Лыткарино", "Люберцы",
     "Москва", "Московский", "Мытищи", "Одинцово", "Подольск", "Пушкино",
     "Реутов", "Химки", "Щербинка", "Щёлково")
)

# create input DataFrame
inputDF = pd.DataFrame(
    {
    'total_square':total_square,
    'floor':floor,
    'rooms':rooms,
    'source_Домклик':source == 'Домклик',
    'source_Новострой-М':source == 'Новострой-М',
    'source_ЦИАН':source == 'ЦИАН',
    'source_Яндекс.Недвижимость':source == 'Яндекс.Недвижимость',
    'city_Балашиха':city == 'Балашиха',
    'city_Видное':city == 'Видное',
    'city_Дзержинский':city == 'Дзержинский',
    'city_Долгопрудный':city == 'Долгопрудный',
    'city_Ивантеевка':city == 'Ивантеевка',
    'city_Королёв':city == 'Королёв',
    'city_Котельники':city == 'Котельники',
    'city_Красногорск':city == 'Красногорск',
    'city_Лобня':city == 'Лобня',
    'city_Лыткарино':city == 'Лыткарино',
    'city_Люберцы':city == 'Люберцы',
    'city_Москва':city == 'Москва',
    'city_Московский':city == 'Московский',
    'city_Мытищи':city == 'Мытищи',
    'city_Одинцово':city == 'Одинцово',
    'city_Подольск':city == 'Подольск',
    'city_Пушкино':city == 'Пушкино',
    'city_Реутов':city == 'Реутов',
    'city_Химки':city == 'Химки',
    'city_Щербинка':city == 'Щербинка',
    'city_Щёлково':city == 'Щёлково'
    },
    index=[0]
)

if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    model_train(train_data)

model = read_model('lr_fitted.pkl')

preds = model.predict(inputDF)
pred = '{0:,}'.format(int(preds)).replace(',', ' ')

#preds = round(preds * 100, 1)

st.image("D:\Python\pythonProject2_HW_8\Gif_image.gif", use_container_width=True)
if st.button("Результат", type="primary"):
    st.write(f"Стоимость Вашей квартиры: {pred} RUB") if preds > 0 else st.write("Невозможно предсказать цену по заданным параметрам.","Пожалуйста, задайте другие параметры и повторите.")