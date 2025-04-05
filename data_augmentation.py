data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)])
# Это часть возможных вариантов.
# RandomFlip("horizontal") — переворачивает по горизонтали
# 50 % случайно выбранных изображений;
# RandomRotation(0.1) — поворачивает входные изображения на случайный угол в
# диапазоне [–10 %, +10 %] (параметр определяет долю полной окружности —
# в градусах заданный здесь диапазон составит [–36, +36]);
# RandomZoom(0.2) — случайным образом изменяет масштаб изображения,
# в данном случае в диапазоне [–20 %, +20 %].

# Это увеличит вариативность входных данных максимум в 197 раз.


# Отображение некоторых дополнительных обучающих изображений.
plt.figure(figsize=(10, 10))
# take (N) позволяет выбрать только N пакетов из набора данных. Этот метод
# действует подобно инструкции break, выполняемой циклом после N-го пакета.
for images, _ in train_dataset.take(1):
    for i in range(9):
        # Применить этап обогащения к пакету изображений/
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        # Вывести первое изображение в выходном пакете. В 9 итерациях будут
        # получены доп. варианты, полученные обогащением 1 и того же фото.
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# Так же как Dropout, слои обогащения изображений неактивны на этапе
# прогнозирования (когда вызывается метод predict() или evaluate()).
# Во время оценки модель # будет вести себя так, как если бы мы не
# задействовали прореживание и обогащение данных.

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)      # добавляем прореживание
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="convnet_from_scratch_with_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")]
history = model.fit(
    train_dataset,
    epochs=100,    # так как много новых вариаций данных (увеличение в 197 раз)
    validation_data=validation_dataset,
    callbacks=callbacks
    )

# Обучение регуляризованной модели.
callbacks = [keras.callbacks.ModelCheckpoint(
    filepath="convnet_from_scratch_with_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")]
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)


# Проверка результатов.
test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

# Благодаря обогащению и прореживанию данных переобучение наступило намного
# позже, в районе 60–70-й эпохи (10 эпохами без обогащения). Точность на
# контрольных данных 81,5%. До обогащения было около 70%.
# Еще бОльший процент даст использование предварительно обученной модели.
