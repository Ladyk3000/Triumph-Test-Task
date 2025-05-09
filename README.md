# Triumph-Test-Task

## Описание
Консольное приложение на Python для автоматического распознавания и разметки персональных данных на третьей странице паспорта РФ. Приложение принимает изображение паспорта (JPG или PNG), выделяет ключевые области (фотография, ФИО, пол, дата рождения, место рождения, серия и номер) и сохраняет результаты в JSON-файле и визуализированном изображении.

## Требования
- Python 3.8+
- Виртуальное окружение (venv или conda)
- Предобученная модель EAST text detector (`frozen_east_text_detection.pb`)

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/Ladyk3000/Triumph-Test-Task
   cd Triumph-Test-Task
   ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Скачайте модель EAST text detector и поместите её в папку `models/`:
   - Ссылка: [frozen_east_text_detection.pb](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/dnn/frozen_east_text_detection.pb)

## Использование
Запустите скрипт, указав путь к изображению:
```bash
python main.py путь_к_изображению.jpg
```
Результаты будут сохранены в папке `output/`:
- JSON-файл с разметкой (`<имя_изображения>_markup.json`)
- Изображение с bounding boxes (`<имя_изображения>_annotated.png`)

## Структура проекта
- `main.py`: Основной скрипт.
- `models/`: Директория для предобученных моделей.
- `output/`: Директория для результатов.
- `requirements.txt`: Зависимости проекта.


## Примечания
- Код соответствует PEP8 (проверено с помощью `flake8` и отформатировано с помощью `black`).
- Для улучшения детекции рекомендуется использовать изображения высокого качества.
- Приложение использует Haar cascades для детекции лиц и EAST text detector для текста.