import matplotlib.pyplot as plt
import numpy as np
import os, shutil, pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.src.utils import image_dataset_from_directory



# Сверточная модель. На вход изображения размером 180*180 с 3 каналами цвета.
inputs = keras.Input(shape=(180, 180, 3))
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
# Выходной слой с вероятностью 0 (кошка) или 1 (собака)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])


# Подготовка данных. Перевод картинок стандартной функцией в тензоры.
# Предполагается, что изображения организованы в подкаталогах, где имя каждого
# подкаталога соответствует классу (например, train/cats/ и train/dogs/).
# Метки проставляются автоматически (в зависимости от каталога).

train_dataset = image_dataset_from_directory(
    # Откуда (из какой папки) брать картинки.
    new_base_dir / "train",
    # Приводим к разрешению 180*180
    image_size=(180, 180),
    # Размер батча. Большие батчи ускоряют вычисления, но требуют больше  ОЗУ.
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)

# Сохраняем модель, если есть улучшение параметров. То есть до переобучения.
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)

# Печать графиков.
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# Тестирование модели.
# Загружаем сохраненную модель (лучший результат ДО переобучения).
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
# Точность около 74%.

# Обогащаем входные данные (поворачиваем, отображаем, масштабируем).

data_augmentation = keras.Sequential(
    # pipeline (конвейер) для преобразования входных данных (изображений).
    # Эти слои работают только во время обучения (когда training=True),
    # при предсказании (inference) они не применяются.
    [
        layers.RandomFlip("horizontal"),    # 50% вероятность горизонт.поворота
        layers.RandomRotation(0.1),         # поворот от -18 до +18 градусов
        layers.RandomZoom(0.2),             # масштабирование ± 20%
    ]
    )

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)               # аугментация
x = layers.Rescaling(1./255)(x)             # нормализация
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)                     # преобразование в вектор
x = layers.Dropout(0.5)(x)                  # регуляризация через Dropout
# На выходе 1 нейрон для бинарной классификации (собака? да\нет)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)

test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
