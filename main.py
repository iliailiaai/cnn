# -*- coding: utf-8 -*-



import logging
import sys


import os
import warnings
#import keras

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import keras
from keras.models import load_model


import gdown
from keras.models import load_model

# URL для скачивания файла модели
url = "https://drive.google.com/uc?id=1_C9mySLYBc6T65wcGDmrnMGA_Jiq4I8t"
output = "cnn_cifar100_model_67.keras"

# Скачиваем модель
gdown.download(url, output, quiet=False)

# Загружаем модель
model = load_model(output)
print("Модель успешно загружена.")


fine_labels = [
    "яблоко", "аквариумная рыбка", "младенец", "медведь", "бобр", "кровать", "пчела", "жук", "велосипед", "бутылка",
    "миска", "мальчик", "мост", "автобус", "бабочка", "верблюд", "банка", "замок", "гусеница", "скот", "стул",
    "шимпанзе", "часы", "облако", "таракан", "диван", "краб", "крокодил", "чашка", "динозавр", "дельфин",
    "слон", "камбала", "лес", "лиса", "девочка", "хомяк", "дом", "кенгуру", "клавиатура", "лампа",
    "газонокосилка", "леопард", "лев", "ящерица", "лобстер", "мужчина", "клен", "мотоцикл", "гора",
    "мышь", "гриб", "дуб", "апельсин", "орхидея", "выдра", "пальма", "груша", "пикап", "сосна", "равнина",
    "тарелка", "мак", "дикобраз", "опоссум", "кролик", "енот", "скат", "дорога", "ракета", "роза", "море",
    "тюлень", "акула", "бурозубка", "скунс", "небоскреб", "улитка", "змея", "паук", "белка", "трамвай",
    "подсолнух", "сладкий перец", "стол", "танк", "телефон", "телевизор", "тигр", "трактор", "поезд", "форель",
    "тюльпан", "черепаха", "гардероб", "кит", "ива", "волк", "женщина", "червь"
]

import telebot
from telebot import types

from telebot import TeleBot, types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Создаем экземпляр бота

#bot = telebot.TeleBot('7552516692:AAGS_5cgcQS3whi3aJ9sb3RAiZ-hhLTo5Hc')
bot = telebot.TeleBot('7943117804:AAFY3_YKcfKAxeV3jZqM9e6RNZLgvBWxyeU')

@bot.message_handler(commands=['start'])
def start(msg: types.Message):

        # Загружаем файл из Dropbox
    try:
        metadata, response = dbx.files_download(folder_path)
        print(f"Файл {metadata.name} успешно загружен.")
    
        # Считываем файл из памяти и загружаем модель
        file_data = BytesIO(response.content)
        model = load_model(file_data)
        print("Модель успешно загружена.")
    except dropbox.exceptions.ApiError as e:
        print(f"Ошибка при загрузке файла: {e}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

    bot.send_message(
        chat_id=msg.chat.id,
        text=f"Привет, {msg.from_user.first_name}!\n\nДанный бот содержит ИИ на конволюционных сверточных сетях (CNN), способный распознавать до 100 типов объектов со средней точностью 67%.",
    )

    bot.send_message(
    chat_id=msg.chat.id,
    text="Для тестирования отправь любую картинку.",
    )

import telebot
import numpy as np
from PIL import Image
from io import BytesIO

# Предобработка изображения
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))

    # Изменение размера до 32x32
    image_resized = image.resize((32, 32))

    # Нормализация пикселей
    image_array = np.array(image_resized).astype('float32') / 255.0

    # Проверка на черно-белое изображение
    if image_array.ndim == 2 or image_array.shape[-1] != 3:
        image_array = np.stack((image_array,) * 3, axis=-1)

    # Добавление измерения батча
    return np.expand_dims(image_array, axis=0)

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def handle_image(msg: telebot.types.Message):
    # Получение файла изображения
    file_info = bot.get_file(msg.photo[-1].file_id)
    file = bot.download_file(file_info.file_path)

    # Предобработка изображения
    image_input = preprocess_image(file)

    # Прогон через модель
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)
    predicted_label = fine_labels[predicted_class]

    # Отправка результата пользователю
    bot.send_message(chat_id=msg.chat.id, text=f"На фото: *{predicted_label}.*\n\nДля повторного тестирования отправь любую картинку еще раз.", parse_mode='Markdown')


# Обработчик ошибок
@bot.message_handler(content_types=['text'])
def handle_text(msg: telebot.types.Message):
    bot.send_message(chat_id=msg.chat.id, text="Это текст, отправь изображение для распознавания.")


#bot.polling(none_stop=True)

# Основной цикл с обработкой ошибок
while True:
    try:
        bot.polling(none_stop=True, interval=1, timeout=20)
    except Exception as e:
        # Логируем ошибку
        logging.error(f"Ошибка в bot.polling: {e}")
