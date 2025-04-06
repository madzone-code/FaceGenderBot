from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)
import os

from predict import (predict_gender, MODELS)

load_dotenv()

IMAGE_PATH = 'photos'
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Создаем инлайн-клавиатуру.
keyboard = [[
    InlineKeyboardButton(key, callback_data=key)] for key in MODELS.keys()]
reply_markup = InlineKeyboardMarkup(keyboard)


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для отпределения пола человека по фотографии. "
        "Отправь мне фото, выбери модель и получи предсказание пола;)"
    )


# Обработчик получения фото.
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if update.message.photo:
        # Получаем фотку с макс.качеством.
        photo = update.message.photo[-1]
        file = await photo.get_file()

    elif update.message.document:
        # Проверяем, является ли документ изображением (фото без сжатия).
        if update.message.document.mime_type.startswith('image/'):
            file = await update.message.document.get_file()

    # Генерируем уникальное имя файла.
    file_path = f"{IMAGE_PATH}/{file.file_id}.jpg"

    # Создаем директорию, если ее нет.
    os.makedirs(IMAGE_PATH, exist_ok=True)

    # Скачиваем фото.
    await file.download_to_drive(file_path)

    # Сохраняем путь к файлу в контексте пользователя.
    context.user_data['photo_path'] = file_path

    # Отправляем сообщение с кнопками
    await update.message.reply_text(
        text='Фото загружено. Выберите модель для предсказания.',
        reply_markup=reply_markup,
        )


# Обработчик нажатия на кнопку.
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Получаем выбранный метод.
    method = query.data
    # Получаем путь к фото из контекста.
    photo_path = context.user_data.get('photo_path')

    if photo_path and os.path.exists(photo_path):
        # Вызываем функцию предсказания.
        prediction = predict_gender(func_name=method, img_path=photo_path)
        # Обработка предсказания.
        gender = "Мужчина" if prediction < 0.5 else "Женщина"
        confidence = (
            prediction if prediction > 0.5 else (1 - prediction)) * 100
        text = (f"Метод {method}.\nПол: {gender}, уверенность: "
                f"{confidence:.2f}%.\nТочность модели на тестовых данных: "
                f"{MODELS[method]}%.")
        # Отправляем результат
        await query.edit_message_text(text=text, reply_markup=reply_markup)


def main():
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    # Фото и фото в виде документа (без сжатия).
    application.add_handler(MessageHandler(
        filters.PHOTO | filters.Document.IMAGE, handle_photo))
    application.add_handler(CallbackQueryHandler(button))
    # Любое текстовое сообщение запускает функцию старт.
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, start))

    # Запускаем бота
    # application.run_polling(allowed_updates=Update.ALL_TYPES)
    application.run_polling()


if __name__ == '__main__':
    main()
