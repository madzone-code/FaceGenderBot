from tensorflow import keras
from keras.src.datasets import mnist
from keras import layers


inputs = keras.Input(shape=(28, 28, 1))
# Задаем количество фильтров и размер окна, которое "скользит" по изображению
# и ищет простые шаблоны (края, линии). 32 фильтра создают 32 "карты" шаблонов.
# На выходе форма (None, 26, 26, 32). 32 - количество "фильтров", 28=32-3+1.
# Чтобы не менялся размер, можно исполььзовать параметр padding = same
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)

# Каждую карту 26×26 уменьшаем в 2 раза, выбирая макс. значение в окне 2×2.
# 26 / 2 = 13 (ширина и высота уменьшаются). Глубина (32) остается прежней.
# На выходе (13, 13, 32).
x = layers.MaxPooling2D(pool_size=2)(x)

# Аналогично. На выходе 64 (филтерс=64), окно 3, поэтому (11, 11, 64)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)

# Аналогично. На выходе (5, 5, 64)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)

# Выравнивание. Все карты (3×3×128 = 1152 значений) "распрямляются" в 1 вектор.
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# Обработка входных данных.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Оценка сверточной сети/
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
