import numpy as np
from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt


MODELS = {"cnn_from_scratch": "73% точность",
          "vgg16_features": "82% точность",
          "vgg16_features_augmentation": "84% точность",
          "fine_tuning": "86% точность",
          }


# Подготовка фото.
def prepare_image(img_path, func_name):
    # Загружаем изображение и изменяем размер
    img = image.load_img(img_path, target_size=(224, 224))
    # Преобразуем в массив numpy
    img_array = image.img_to_array(img)

    # Для повышения точности мы нормализовали пиксели в cnn_from_scratch.
    # # для предсказания делаем то же, но только у cnn_from_scratch.
    # if func_name == 'cnn_from_scratch':
    #     # Нормализуем значения пикселей в диапазон [0, 1]
    #     img_array = img_array / 255.0
    # Добавляем размер батча (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def prepare_vgg16_features(prepared_img):
    # Загрузка предобученной VGG16 без верхних слоев
    conv_base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = False  # Замораживаем веса
    # Предобработка для VGG16
    preprocessed_img = keras.applications.vgg16.preprocess_input(prepared_img)
    # Извлечение признаков через conv_base
    features = conv_base.predict(preprocessed_img)  # Форма: (1, 7, 7, 512)
    # Возвращаем извлеченные VGG16 признаки из фото.
    return features


def predict_gender(func_name, img_path):
    # Загрузка нужной модели.
    model = keras.models.load_model(f'models/{func_name}.keras')
    # Подготовка изображения
    prepared_img = prepare_image(img_path, func_name)

    # В случае с vgg16_features предсказание не по фото,
    # а по извелеченным из него признакам
    if func_name == "vgg16_features":
        prepared_img = prepare_vgg16_features(prepared_img)

    # Предсказание
    prediction = model.predict(prepared_img)

    # Интерпритация результата.
    pred_class = "мужчина" if prediction[0][0] < 0.5 else "женщина"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    # Вывод результата
    print(f"Предсказанный класс: {pred_class}")
    print(f"Уверенность: {confidence:.3f}")
    print(f"Модель: {func_name}, точность на тестовых данных {MODELS[func_name]}")

    return (f"Предсказанный класс: {pred_class}"
            f"Уверенность: {confidence:.3f}"
            f"Модель: {func_name}, точность на тестовых данных {MODELS[func_name]}"
            )


if __name__ == "__main__":
    img_path = "photos/3.jpg"
    for model_name in MODELS.keys():
        predict_gender(model_name, img_path)
        print()
