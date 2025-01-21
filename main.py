"""
В этом задании вам предстоит обучить модель для прогнозирования стоимости недвижимости и создать API для модели с помощью FastAPI.

Вам нужно:
Скачать данные и обучить модель прогнозирования стоимости недвижимости (модель может быть любой сложности,
даже линейная регрессия на двух признаках. Можно переиспользовать модель с прошлого ДЗ) - 1 балл +
Реализуйте код для получения предсказания обученной моделью - 1 балл +
Реализуйте получение предсказания моделью через get-запрос по адресу /predict_get - 2 балла +
Реализуйте получение предсказания моделью через post-запрос по адресу /predict_post - 2 балла +
Реализуйте liveness-пробу (health-check) health - 1 балл +
Запустите API через uvicorn, посетите адрес http://127.0.0.1:8000/docs и попробуйте отправить get/post запросы к вашему API
через интерфейс Swagger-документации.
Результаты выполнения запросов сохраните в виде скриншотов в репозитории - 3 балла +
"""


import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("lr_fitted_2.pkl", 'rb') as file:
    model = joblib.load(file)


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

# post-запрос
class ModelRequestData(BaseModel):
    total_square: int
    floor: int
    rooms: int
    source_Домклик: bool
    source_ЦИАН: bool
    city_Балашиха: bool
    city_Видное: bool
    city_Дзержинский: bool
    city_Долгопрудный: bool
    city_Ивантеевка: bool
    city_Королёв: bool
    city_Котельники: bool
    city_Красногорск: bool
    city_Лобня: bool
    city_Лыткарино: bool
    city_Люберцы: bool
    city_Москва: bool
    city_Московский: bool
    city_Мытищи: bool
    city_Одинцово: bool
    city_Подольск: bool
    city_Пушкино: bool
    city_Реутов: bool
    city_Химки: bool
    city_Щербинка: bool
    city_Щёлково: bool
    source_НовостройМ: bool
    source_ЯндексНедвижимость: bool


@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

# get-запрос
test_query = {
    "total_square": 70,
    "floor": 1,
    "rooms": 2,
    "source_Домклик": False,
    "source_ЦИАН": True,
    "city_Балашиха": False,
    "city_Видное": False,
    "city_Дзержинский": False,
    "city_Долгопрудный": False,
    "city_Ивантеевка": False,
    "city_Королёв": False,
    "city_Котельники": False,
    "city_Красногорск": False,
    "city_Лобня": False,
    "city_Лыткарино": False,
    "city_Люберцы": False,
    "city_Москва": True,
    "city_Московский": False,
    "city_Мытищи": False,
    "city_Одинцово": False,
    "city_Подольск": False,
    "city_Пушкино": False,
    "city_Реутов": False,
    "city_Химки": False,
    "city_Щербинка": False,
    "city_Щёлково": False,
    "source_НовостройМ": False,
    "source_ЯндексНедвижимость": False
}

@app.get("/predict_get", response_model=Result)
def preprocess_data(data):
    input_data = test_query  # .dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)