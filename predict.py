import numpy as np
from tensorflow import keras
from keras.preprocessing import image


# из обучающей модели.
WEIGHT, HEIGHT = 224, 224

MODELS = {"cnn_from_scratch": 73,
          "vgg16_features": 82,
          "vgg16_features_augmentation": 84,
          "fine_tuning": 86,
          }


# Подготовка фото.
def prepare_image(img_path):
    # Загружаем изображение и изменяем размер
    img = image.load_img(img_path, target_size=(224, 224))
    # Преобразуем в массив numpy
    img_array = image.img_to_array(img)
    # Добавляем batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def prepare_vgg16_features(prepared_img):
    # Загрузка предобученной VGG16 без верхних слоев
    conv_base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(WEIGHT, HEIGHT, 3))
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
    prepared_img = prepare_image(img_path)
    # В случае с vgg16_features предсказание не по фото,
    # а по извелеченным из него признакам.
    if func_name == "vgg16_features":
        prepared_img = prepare_vgg16_features(prepared_img)

    # Предсказание
    prediction = model.predict(prepared_img)

    return prediction[0][0]


if __name__ == "__main__":
    for model in MODELS.keys():
        for i in range(1, 11):
            prediction = predict_gender(f"{model}", f"photos/{i}.jpg")
            print('баба' if prediction > 0.5 else 'мужик')

        print(f"Это была модель {model}")
        print()
