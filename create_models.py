import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.src.utils import image_dataset_from_directory

from prepare_input import split_dataset


# Определяем формат входных данных.
WEIGHT, HEIGHT = 224, 224                               # родное для vgg16


# Подготавливаем тензоры.
def create_tensors(path):
    train_dataset = image_dataset_from_directory(
        # Откуда (из какой папки) брать картинки.
        path / "train",
        # Приводим к разрешению weight, height (224*224)
        image_size=(WEIGHT, HEIGHT),
        # Размер батча. Большие ускоряют вычисления, но требуют больше ОЗУ.
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        path / "val",
        image_size=(WEIGHT, HEIGHT),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        path / "test",
        image_size=(WEIGHT, HEIGHT),
        batch_size=32)
    return train_dataset, validation_dataset, test_dataset


# Создание экземпляра сверточной основы VGG16.
conv_base = keras.applications.vgg16.VGG16(
    # Источник весов для инициализации модели
    weights="imagenet",
    # Необходимость подключения к сети полносвязного классификатора
    # (пределяет принадлежность изображения к 1000 классов), не подключаем.
    include_top=False,
    # Не обязательно использовать. Без него можно работать с изображениями
    # любых размеров.
    input_shape=(WEIGHT, HEIGHT, 3)
    )

conv_base.summary()                     # просмотр инфы по сверточной основе


# Сверточная сеть с нуля. Аугментация.
def cnn_from_scratch(train_dataset, validation_dataset, test_dataset):

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

    # Сверточная модель. На вход изображения размером 224*224 с 3 цветами.
    inputs = keras.Input(shape=(WEIGHT, HEIGHT, 3))
    # Аугментация.
    x = data_augmentation(inputs)
    # Нормализация.
    x = layers.Rescaling(1./255)(x)  # нормализация данных в диапазон [0, 1]
    # 5 блоков свертки и пулинга. Количество фильтров увеличивается: 32 -> 256
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # Преобразуем тензор (7, 7, 256) в плоский вектор (12544). 7*7*256 = 12544.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    # Выходной слой с вероятностью 0 (мужчина) или 1 (женщина).
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    callbacks = [
        # Сохраняем модель, если есть улучшение параметров.
        keras.callbacks.ModelCheckpoint(
            filepath="models/cnn_from_scratch.keras",
            save_best_only=True,
            monitor="val_loss"),
        # Прерываем обучение после 7 эпох без улучшений.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True),
        ]

    history = model.fit(
        train_dataset,
        epochs=100,              # с запасом, EarlyStopping отработает
        validation_data=validation_dataset,
        callbacks=callbacks)

    plot_results(history)       # для настройки, отключить EarlyStopping

    # Тестирование модели.
    # Загружаем сохраненную модель (лучший результат, ДО переобучения).
    best_model = keras.models.load_model("models/cnn_from_scratch.keras")
    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Test accuracy: {test_acc:.3f}")

    return best_model
# 77% точность.


# Печатаем графики
def plot_results(history):
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(12, 5))  # Устанавливаем размер окна
    # График потерь
    plt.subplot(1, 2, 1)  # Первое подокно
    plt.plot(epochs, history_dict["loss"], "bo", label="Потери на обучении")
    plt.plot(epochs, history_dict["val_loss"], "b", label="Потери на проверке")
    plt.title("Потери на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    # График точности
    plt.subplot(1, 2, 2)  # Второе подокно
    plt.plot(epochs, history_dict["accuracy"], "bo",
             label="Точность на обучении")
    plt.plot(epochs, history_dict["val_accuracy"], "b",
             label="Точность на проверке")
    plt.title("Точность на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()

    plt.tight_layout()  # Улучшаем расположение графиков
    plt.show()


# Быстрое выделение признаков без обогащения данных. Быстро и дешево.
def vgg16_features(conv_base, train_dataset, validation_dataset, test_dataset):

    # Получение признаков и меток из фото.
    def get_features_and_labels(dataset):
        all_features = []       # список для хранения признаков изображений
        all_labels = []         # список для хранения соответствующих меток
        for images, labels in dataset:
            # Подготавливает изображения для VGG16.
            preprocessed_images = (
                keras.applications.vgg16.preprocess_input(images))
            # conv_base - извлекает признаки из изображений.
            # Пропускает предобработанные изображения через модель и возвращает
            # признаки (тензор размером (batch_size, height, width, channels).
            # Признаки - высокоуровневые представления изображений,
            # извлеченные сверточной сетью.
            features = conv_base.predict(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)
        return np.concatenate(all_features), np.concatenate(all_labels)

    # Все датасеты преобразовываем в признаки и метки.
    train_features, train_labels = (
        get_features_and_labels(train_dataset))
    val_features, val_labels = (
        get_features_and_labels(validation_dataset))
    test_features, test_labels = (
        get_features_and_labels(test_dataset))

    # Определение и обучение полносвязного классификатора, используя vgg16.
    # То есть мы получаем представления изображений,
    # а потом передаем их на вход новой моделе.
    # Создаем входной тензор 5,5,512 (если 180*180) (без учета размера пакета).
    inputs = keras.Input(shape=(7, 7, 512))
    # "Расплющиваем" входной тензор в вектор.
    x = layers.Flatten()(inputs)
    # Уменьшаем размерность данных и извлекает более абстрактные признаки.
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    # Слой для бинарной классификации.
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/vgg16_features.keras",
            save_best_only=True,
            monitor="val_loss")
        ]

    history = model.fit(
        train_features, train_labels,
        epochs=20,
        validation_data=(val_features, val_labels),
        callbacks=callbacks)

    plot_results(history)

    best_model = keras.models.load_model(
        "models/vgg16_features.keras")
    test_acc = best_model.evaluate(test_features, test_labels)[1]

    print(f"Test accuracy: {test_acc:.3f}")

    return best_model
# 82% точность.


# Выделение признаков с обогащением данных.
def vgg16_features_augmentation(
        conv_base, train_dataset, validation_dataset, test_dataset):

    # При передаче trainable=False веса слоев замораживаются (не обучаются).
    conv_base.trainable = False

    # Теперь создадим новую модель, объединяющую обогащение данных,
    # замороженную сверточную основу, полносвязанный классификатор.

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )

    inputs = keras.Input(shape=(WEIGHT, HEIGHT, 3))
    x = data_augmentation(inputs)                     # обогащение данных
    x = keras.applications.vgg16.preprocess_input(x)  # масштабирование входных
    x = conv_base(x)
    x = layers.Flatten()(x)
    # Обучаются веса только 2 слоев Dense.
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/vgg16_features_augmentation.keras",
            save_best_only=True,
            monitor="val_loss"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True)
        ]

    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks)

    plot_results(history)

    best_model = keras.models.load_model(
        "models/vgg16_features_augmentation.keras")

    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Test accuracy: {test_acc:.3f}")

    return best_model
# 84% точность


def fine_tuning(conv_base, train_dataset, validation_dataset, test_dataset):
    # Дообучение(fine-tuning) предварительно обученной модели. Для этого нужно:
    # - добавить свою сеть поверх обученной базовой сети
    # - заморозить базовую сеть
    # - обучить добавленную часть
    # - разморозить несколько слоев в базовой сети
    # - обучить эти слои и добавленную часть вместе
    # Причем, переобучаем только слои с крупными (обобщенными) признаками.
    # Первые 3 этапа мы уже сделали в vgg16_features_augmentation.

    # Получаем предварительную модель
    model = vgg16_features_augmentation(
        conv_base, train_dataset, validation_dataset, test_dataset)

    # Рамораживаем все слои, кроме последних 4.
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/fine_tuning.keras",
            save_best_only=True,
            monitor="val_loss"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True)
    ]

    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Визуализация результатов обучения.
    plot_results(history)

    # Загружаем лучшую модель и оцениваем на тестовом наборе
    best_model = keras.models.load_model("models/fine_tuning.keras")
    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Точность на тестовом наборе: {test_acc:.3f}")

    return best_model
# 86% точность.


if __name__ == '__main__':
    # Подготавливаем исходные данные (3 набора: микро, маленький, большой).
    path_to_micro = split_dataset('micro_dataset', 1000, 300, 1000)
    path_to_small = split_dataset('small_dataset', 3000, 1000, 3000)
    path_to_big = split_dataset('big_dataset', 10000, 3000, 10000)

    # Подготавливает 3 вида тензоров по исходным данным.
    train_micro_dataset, validation_micro_dataset, test_micro_dataset = (
        create_tensors(path_to_micro))
    train_small_dataset, validation_small_dataset, test_small_dataset = (
        create_tensors(path_to_small))
    train_big_dataset, validation_big_dataset, test_big_dataset = (
        create_tensors(path_to_big))

    # Создаем 4 модели.
    cnn_from_scratch(
        train_small_dataset, validation_small_dataset, test_small_dataset)
    vgg16_features(conv_base, train_small_dataset,
                   validation_small_dataset, test_small_dataset)
    vgg16_features_augmentation(conv_base, train_small_dataset,
                                validation_small_dataset, test_small_dataset)
    fine_tuning(conv_base, train_small_dataset,
                validation_small_dataset, test_small_dataset)
