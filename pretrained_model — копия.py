import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.src.utils import image_dataset_from_directory

from prepare_input import split_dataset


# Подготавливаем исходные данные (3 набора: микро, маленький, большой).
path_to_micro = split_dataset('micro_dataset', 100, 30, 100)
path_to_small = split_dataset('small_dataset', 1000, 300, 1000)
path_to_big = split_dataset('big_dataset', 8000, 2000, 8000)

# Определяем формат входных данных.
weight, height = 224, 224


# Подготавливаем тензоры.
def create_tensors(path):
    train_dataset = image_dataset_from_directory(
        # Откуда (из какой папки) брать картинки.
        path / "train",
        # Приводим к разрешению weight, height (224*224)
        image_size=(weight, height),
        # Размер батча. Большие ускоряют вычисления, но требуют больше ОЗУ.
        batch_size=32)
    validation_dataset = image_dataset_from_directory(
        path / "val",
        image_size=(weight, height),
        batch_size=32)
    test_dataset = image_dataset_from_directory(
        path / "test",
        image_size=(weight, height),
        batch_size=32)
    return train_dataset, validation_dataset, test_dataset


train_micro_dataset, validation_micro_dataset, test_micro_dataset = (
    create_tensors(path_to_micro))
train_small_dataset, validation_small_dataset, test_small_dataset = (
    create_tensors(path_to_small))
train_big_dataset, validation_big_dataset, test_big_dataset = (
    create_tensors(path_to_big))

# Создание экземпляра сверточной основы VGG16
conv_base = keras.applications.vgg16.VGG16(
    # Источник весов для инициализации модели
    weights="imagenet",
    # Необходимость подключения к сети полносвязного классификатора
    # (пределяет принадлежность изображения к 1000 классов), не подключаем.
    include_top=False,
    # Не обязательно использовать. Без него можно работать с изображениями
    # любых размеров.
    input_shape=(weight, height, 3)
    )

conv_base.summary()             # просмотр инфы по сверточной основе


def cnn_from_scratch(train_dataset, validation_dataset, test_dataset):
    # Сверточная модель. На вход изображения размером 224*224 с 3 каналами цвета.
    inputs = keras.Input(shape=(weight, height, 3))
    # Нормализуем значения пикселей из [0, 255] в [0, 1] (делим на 255).
    x = layers.Rescaling(1./255)(inputs)
    # 5 блоков свертки и пулинга. Количество фильтров увеличивается: от 32 до 256.
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # Преобразуем тензор (7, 7, 256) в плоский вектор (12544). 7 * 7 * 256 = 12544.
    x = layers.Flatten()(x)
    # Выходной слой с вероятностью 0 (женщина) или 1 (мужчина).
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

    callbacks = [
        # Сохраняем модель, если есть улучшение параметров.
        keras.callbacks.ModelCheckpoint(
            filepath="cnn_from_scratch.keras",
            save_best_only=True,
            monitor="val_loss"),
        ]

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

    plot_results(history)

    # Тестирование модели.
    # Загружаем сохраненную модель (лучший результат ДО переобучения).
    best_model = keras.models.load_model("cnn_from_scratch.keras")
    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Test accuracy: {test_acc:.3f}")
    # 67% на маленьком датасете, 77% на большом.

    return best_model


# Печатаем графики
def plot_results(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Точность на этапе обучения")
    plt.plot(epochs, val_acc, "b", label="Точность на этапе проверки")
    plt.title("Точность на этапах обучения и проверки")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Потери на этапе обучения")
    plt.plot(epochs, val_loss, "b", label="Потери на этапе проверки")
    plt.title("Потери на этапах обучения и проверки")
    plt.legend()
    plt.show()


# Быстрое выделение признаков без обогащения данных. Быстро и дешево.
def vgg16_features(conv_base, train_dataset, validation_dataset, test_dataset):

    def get_features_and_labels(dataset):
        all_features = []       # список для хранения признаков изображений
        all_labels = []         # список для хранения соответствующих меток
        for images, labels in dataset:
            # Подготавливает изображения для VGG16.
            preprocessed_images = keras.applications.vgg16.preprocess_input(images)
            # conv_base - предобученная модель, извлекает признаки из изображений.
            # Пропускает предобработанные изображения через модель и возвращает
            # признаки (тензор размером (batch_size, height', width', channels').
            # Признаки - высокоуровневые представления изображений,
            # извлеченные сверточной сетью.
            features = conv_base.predict(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)
        return np.concatenate(all_features), np.concatenate(all_labels)

    train_features, train_labels = (
        get_features_and_labels(train_dataset))
    val_features, val_labels = (
        get_features_and_labels(validation_dataset))
    test_features, test_labels = (
        get_features_and_labels(test_dataset))

    # Определение и обучение полносвязного классификатора.
    # Создаем входной тензор 7,7,512 (без учета размера пакета).
    inputs = keras.Input(shape=(7, 7, 512))
    # "Расплющиваем" входной тензор в вектор.
    x = layers.Flatten()(inputs)
    # Уменьшает размерность данных и извлекает более абстрактные признаки.
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
            filepath="vgg16_features.keras",
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
        "vgg16_features.keras")
    test_acc = best_model.evaluate(test_features, test_labels)[1]

    print(f"Test accuracy: {test_acc:.3f}")
    # 79% на маленьком датасете.

    return best_model


# Выделение признаков с обогащением данных.
def vgg16_features_augmentation(conv_base, train_dataset, validation_dataset, test_dataset):

    # При передаче trainable=False список обучаемых весов слоя/модели очищается.
    conv_base.trainable = False

    # Теперь создадим новую модель, объединяющую обогащение данных, замороженную
    # сверточную основу, полносвязанный классификатор.

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ]
    )

    inputs = keras.Input(shape=(weight, height, 3))
    x = data_augmentation(inputs)                       # обогащение данных
    x = keras.applications.vgg16.preprocess_input(x)    # масштабирование входных
    x = conv_base(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="vgg16_features_augmentation.keras",
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
        "vgg16_features_augmentation.keras")
    
    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Test accuracy: {test_acc:.3f}")
    # 82% на маленьком датасете.

    return best_model


def fine_tuning(conv_base, train_dataset, validation_dataset, test_dataset):
    # Дообучение(fine-tuning) предварительно обученной модели. Для этого нужно:
    # - добавить свою сеть поверх обученной базовой сети
    # - заморозить базовую сеть
    # - обучить добавленную часть
    # - разморозить несколько слоев в базовой сети
    # - обучить эти слои и добавленную часть вместе
    # Причем, переобучаем только слои с крупными (обобщенными) признаками.

    # Замораживаем базовую модель
    conv_base.trainable = False

    # Обогащение данных
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        ])

    # Определяем входной слой
    inputs = keras.Input(shape=(weight, height, 3))
    x = data_augmentation(inputs)  # Добавляем обогащение данных
    x = keras.applications.vgg16.preprocess_input(x)  # Предобработка для VGG16
    x = conv_base(x)  # Пропускаем через базовую модель
    x = layers.Flatten()(x)  # Преобразуем в одномерный вектор
    x = layers.Dense(256)(x)  # Полносвязный слой
    x = layers.Dropout(0.7)(x)  # Увеличиваем Dropout до 0.7
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Выходной слой для бинарной классификации
    model = keras.Model(inputs, outputs)

    # Компилируем модель
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    # Callback для сохранения лучшей модели
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning_step1.keras",
            save_best_only=True,
            monitor="val_loss"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True)
    ]

    # Первый этап обучения (с замороженной базой)
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Визуализация результатов.
    plot_results(history)

    # Размораживаем базовую модель для тонкой настройки
    conv_base.trainable = True
    # Замораживаем все слои, кроме последних 4
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    # Перекомпилируем модель с меньшей скоростью обучения
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-6),
                  metrics=["accuracy"])

    # Callback для второго этапа
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="fine_tuning_step2.keras",
            save_best_only=True,
            monitor="val_loss"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True)
    ]

    # Второй этап обучения (тонкая настройка)
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset,
        callbacks=callbacks)

    # Визуализация результатов второго этапа
    plot_results(history)

    # Загружаем лучшую модель и оцениваем на тестовом наборе
    best_model = keras.models.load_model("fine_tuning_step2.keras")
    test_acc = best_model.evaluate(test_dataset)[1]
    print(f"Точность на тестовом наборе: {test_acc:.3f}")
    # 80% на маленьком датасете, 88% на большом.
    return best_model


# cnn_from_scratch(train_small_dataset, validation_small_dataset, test_small_dataset)
# vgg16_features(conv_base, train_small_dataset, validation_small_dataset, test_small_dataset)
# vgg16_features_augmentation(conv_base, train_small_dataset, validation_small_dataset, test_small_dataset)
fine_tuning(conv_base, train_small_dataset, validation_small_dataset, test_small_dataset)
fine_tuning(conv_base, train_big_dataset, validation_big_dataset, test_big_dataset)


"""
# Загрузка моделей
features = keras.models.load_model("vgg16_features.keras")
features_augmentation = keras.models.load_model("vgg16_features_augmentation.keras")
fine_tuning_model = keras.models.load_model("fine_tuning_step2.keras")

# Извлечение признаков для модели features
test_small_features, test_small_labels = get_features_and_labels(test_small_dataset)


# Оценка модели features на извлечённых признаках
test_acc = features.evaluate(test_small_features, test_small_labels)[1]
print(f"Точность метода features: {test_acc:.3f}")

# Оценка модели features_augmentation на исходных изображениях
test_acc = features_augmentation.evaluate(test_small_dataset)[1]
print(f"Точность метода features_augmentation: {test_acc:.3f}")

# Оценка модели fine_tuning на исходных изображениях (для большого набора данных)
test_acc = fine_tuning_model.evaluate(test_small_dataset)[1]  # SMALL!!!
print(f"Точность метода fine_tuning: {test_acc:.3f}")
"""