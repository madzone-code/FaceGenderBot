import os
import shutil
from pathlib import Path
import random


def split_dataset(dir_name, train_count, val_count, test_count):
    """
    Распределяет файлы по наборам train, val, test с разделением по полу.

    Args:
        dir_name: str - имя директории, куда будут сложены обработанные фото.
        train_count: int - общее количество примеров для обучения
        val_count: int - общее количество примеров для валидации
        test_count: int - общее количество примеров для теста

    Return:
        base_dir - путь созданной директории, где лежат подготовленные данные.
    """
    # Исходная директория с файлами
    original_dir = Path("data/UTKFace")         # исходная папка с фотками
    base_dir = Path(f"data/{dir_name}")    # папка с обработанными фото

    if base_dir.exists():
        print('Выходная папка уже существует. Прерываем выполнение.')
        return base_dir

    # Список всех файлов с расширением .jpg
    all_files = list(original_dir.glob("*.jpg"))
    # Перемешиваем список файлов
    random.shuffle(all_files)

    # Отделяем файлы по полу (создаем 2 категории).
    women_files = []
    men_files = []

    # Исходные имена вида [age]_[gender]_[race]_[date&time].jpg. Нас интересует
    # пол (второй элемент). 0 - мужчина, 1 - женщина.
    for fname in all_files:
        # Извлекаем имя файла без расширения, делим его на части по символу '_'
        parts = fname.stem.split('_')
        gender = parts[1]
        if gender == '1':
            women_files.append(fname)
        elif gender == '0':
            men_files.append(fname)

    # Определяем наборы данных
    datasets = {
        "train": train_count,
        "val": val_count,
        "test": test_count
    }

    # Обрабатываем каждый набор данных
    for subset_name, total_count in datasets.items():
        subset_dir = base_dir / subset_name
        # Вычисляем количество для каждой категории (половины от общего)
        category_count = total_count // 2

        for category in ("women", "men"):
            dir = subset_dir / category
            os.makedirs(dir, exist_ok=True)

            # Выбираем файлы для текущей категории
            source_files = women_files if category == "women" else men_files
            # Берем нужное количество файлов
            selected_files = source_files[:category_count]

            # Копируем файлы с новыми именами
            for i, fname in enumerate(selected_files):
                new_fname = f"{category}_{i}.jpg"
                shutil.copyfile(src=fname, dst=dir / new_fname)

            # Удаляем использованные файлы из списка
            if category == "women":
                women_files = women_files[category_count:]
            else:
                men_files = men_files[category_count:]
    # Возвращаем путь вновь созданной директории, где лежат данные.
    return base_dir


if __name__ == "__main__":
    split_dataset('small_dataset', 1000, 300, 1000)
    split_dataset('big_dataset', 10000, 3000, 10000)
