model = get_mnist_model()
model.compile(optimizer="rmsprop",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

tensorboard = keras.callbacks.TensorBoard(
    log_dir="./logs",   # Указываем путь к директории логов
    histogram_freq=1,   # Частота записи гистограмм (например, каждую эпоху)
    write_graph=True,   # Записывать граф модели
    write_images=True,  # Записывать изображения весов (если применимо)
)

model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(val_images, val_labels),
          callbacks=[tensorboard])
# После обучения можем запустить сервер:
# tensorboard --logdir ./logs
# Если обучение производится в блокноте Colab, то можно запустить встроенный
# экземпляр TensorBoard в блокноте, выполнив следующую команду:
# %load_ext tensorboard
# %tensorboard --logdir /full_path_to_your_log_dir