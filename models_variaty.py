from tensorflow import keras
from keras import layers
import numpy as np

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")],
    name='Testik')
# Пока весов еще нет, появляются после построения.

# model = keras.Sequential(name="testik")
# model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
# model.add(layers.Dense(10, activation="softmax"))

model.build(input_shape=(None, 3))  # None - размер любой
# print(model.weights)              # печать весов

# Краткая информация о структуре модели: имена слоёв, их типы, размеры выходных
# данных и количество параметров.
model.summary()

# Создаем ту же модель, но руками.
# Модель принимает любое количество образцов, длина каждого (3,)
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)      # вход слоя-inputs
outputs = layers.Dense(10, activation="softmax")(features)  # вход - features
model_2 = keras.Model(inputs=inputs, outputs=outputs)

model_2.summary()


# 3. Функциональный API.
# Задача. Создать модель, 3 входа, 2 выхода.
vocabulary_size = 10000     # количество слов для классификации
num_tags = 100              # количество тегов
num_departments = 4         # количество департаментов

# Определение входов модели.
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# Объединение входных признаков в один тензор features.
features = layers.Concatenate()([title, text_body, tags])
# Добавление промежуточного слоя для рекомбинации входных признаков в более
# богатые представления.
features = layers.Dense(64, activation="relu", name='manual_layer')(features)

# Определение выходов модели.
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department")(features)

# Создание модели с передачей ей информации о входах и выходах.
# features - внутренний тензор, часть вычислительного графа, который соединяет
# входные слои (title, text_body, tags) с выходными (priority, department).
model_manual = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department],
    name='pupsik',
    )
model_manual.summary()

# Обучение модели с передачей массивов входных данных и целей.
num_samples = 1280

# Фиктивные входные данные.
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# Фиктивные целевые данные.
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

# Обучаем так.
model_manual.compile(optimizer="rmsprop",
                     loss=["mean_squared_error", "categorical_crossentropy"],
                     metrics=[["mean_absolute_error"], ["accuracy"]])
model_manual.fit([title_data, text_body_data, tags_data],
                 [priority_data, department_data],
                 epochs=1)
model_manual.evaluate([title_data, text_body_data, tags_data],
                      [priority_data, department_data])
priority_preds, department_preds = model_manual.predict(
    [title_data, text_body_data, tags_data])

# Или так. на выбор.
model_manual.compile(optimizer="rmsprop",
                     loss={"priority": "mean_squared_error",
                           "department": "categorical_crossentropy"},
                     metrics={"priority": ["mean_absolute_error"],
                              "department": ["accuracy"]})
model_manual.fit({
    "title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data},
    epochs=1)
model_manual.evaluate(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data},
    {"priority": priority_data, "department": department_data})

# Сохранение картинки со структурой модели в той папке, откуда запускается.
keras.utils.plot_model(model_manual,
                       "updated_ticket_classifier.png",
                       show_shapes=True)


# 4.Создание производного подкласса класса Model.
class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        # Не забываем вызывать родительский конструктор.
        super().__init__()
        # Определяем слои в конструкторе.
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax")

    # Определяем порядок выполнения прямого прохода в методе call().
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


model = CustomerTicketModel(num_departments=4)
priority, department = model(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data})

model.compile(optimizer="rmsprop",
              # Структура аргументов, передаваемых в параметрах loss и metrics,
              # должна соответствовать тому, что возвращает call(), здесь - это
              # списки с двумя элементами.
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])

tensorboard = keras.callbacks.TensorBoard(
    log_dir="./logs",   # Указываем путь к директории логов
    histogram_freq=1,   # Частота записи гистограмм (например, каждую эпоху)
    write_graph=True,   # Записывать граф модели
    write_images=True,  # Записывать изображения весов (если применимо)
)

# Структура входных данных должна точно соответствовать структуре параметров
# метода call(), здесь это словарь с ключами title, text_body и tags.
model.fit({"title": title_data,
           "text_body": text_body_data,
           "tags": tags_data},
          # Структура цели должна точно соответствовать возврату call().
          [priority_data, department_data],
          epochs=1,

          callbacks=[tensorboard])

model.evaluate({"title": title_data,
                "text_body": text_body_data,
                "tags": tags_data},
               [priority_data, department_data])
priority_preds, department_preds = model.predict({"title": title_data,
                                                  "text_body": text_body_data,
                                                  "tags": tags_data})

