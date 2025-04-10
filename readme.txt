Архитектура проекта:

Возможные решения PyTessaract, EasyOCR, PaddleOCR, DocTR. Выбор PyTessaract так как есть traindata для работы с кирилецей.
Остальные допускают 20+ % ошибок с распознованием кирилицы.


1. Pytesseract OCR (Оптическое Распознавание Символов):
    (Использование OCR-технологии для извлечения текста из сканированных документов
    Google Cloud Vision API имеет под капотом tesseract так как в 2018г Google выкупила tesseract.
    Поэтой причине целесообразно использовать локальный дистрибутив tesseract-ocr-w64-setup-v5.3.0.20221214)

2. Предобработка текста:
    - Очистка и нормализация текста.
    - Инструменты: Для нормализация текста (удаление спецсимволов, исправление ошибок).
    - Библиотеки:
        PySpellChecker,
        Hunspell,
        LanguageTool,
        DeepPavlov.

3. Создание векторной базы данных:
    - Преобразование текста в векторное представление.
    - Инструменты: Использование моделей эмбеддинга и библиотек для векторного поиска FAISS.
    - Система RAG (Retrieval-Augmented Generation) для извлечения 100% верной информации для ответа.
    - LLM: gpt-4o-mini в сочетании с векторной базой для реализации RAG. Модель копеечная и на тестах за пол года ялвяется лидером.

4. Пользовательский интерфейс для взаимодействия с системой.
    - Веб-приложение:
        FastAPI, ТГ бот.


Схема проетка: 

                +----------------------+\
                \|    Пользователь      |\
                +----------+-----------+\
                |\
                △ Пользователь загружает PDF\
                |\
                +----------v-----------+\
                \|    Обработка FastAPI   |\
                +----------+-----------+\
                |\
                □ Извлечение текста\
                |\
                +----------v-----------+\
                \|  Предобработка текста |\
                +----------+-----------+\
                |\
                □ Исправление ошибок\
                |\
                +----------v-----------+\
                \|  Создание векторной БД  |\
                +----------+-----------+\
                |\
                □ Векторизация\
                |\
                +----------v-----------+\
                \|  Уведомление Telegram   |\
                +----------+-----------+\
                |\
                □ Отправка уведомления\
                |\
                +----------v-----------+\
                \|   Взаимодействие с LLM в боте ТГ |\
                +----------+-----------+\
                |\
                □ Обработка запросов\
                |\
                +----------v-----------+\
                \|  Обработка новых сканов |\
                +----------+-----------+\
                |\
                △ Обновление векторной БД\
                |\
                +----------v-----------+\
                \|    Завершение процесса  |\
                +----------------------+



✅ Зависимости
Убедитесь, что у вас установлены:

Python 3.10+

Tesseract OCR (локально установлен)
Poppler for Windows (локально установлен)

Tesseract: https://github.com/tesseract-ocr/tesseract
Poppler: https://github.com/oschwartz10612/poppler-windows/releases

📦 Установка
Клонируйте проект и создайте виртуальное окружение:


git clone https://github.com/your_username/OCR_bot.git
cd OCR_bot
python -m venv .venv
.\.venv\Scripts\activate

Установите зависимости:

pip install -r requirements.txt

⚙️ Настройка переменных окружения

Создайте файл .env в корне проекта и пропишите туда пути:
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\Program Files (x86)\poppler-24.08.0\Library\bin
OPENAI_API_KEY=
URL=https://api.openai.com/v1/chat/completions
SENTENCE_TRANSFORMERS_COS_V1=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
SENTENCE_TRANSFORMERS_BASE_V2=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
url=https://api.telegram.org/ваш_токен/getUpdates
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

SMTP_HOST=smtp.gmail.com
SMTP_PORT=
SMTP_USER=
SMTP_PASS=

REDIS_HOST=localhost
EDIS_PORT=6379

TAVILY_KEY=


🚀 Запуск
bash

uvicorn app.main:app --reload
После запуска API будет доступно по адресу:
http://127.0.0.1:8000

Swagger-документация:
http://127.0.0.1:8000/docs

📤 Поддерживаемые типы загрузки:

single	.png, .jpg, .jpeg	Одиночное изображение сканы
archive	.zip	Архив с изображениями
pdf	.pdf	PDF-документ с изображениями страниц
📌 Пример запроса через Swagger
Перейдите на /docs

Нажмите "Try it out"

Выберите файл

Укажите fileType — один из single, archive, pdf, скан

Нажмите Execute

🧪 Тестовые кейсы
📄 PDF-файл скана

🖼️ PNG с текстом


🛠️ Отладка
Если при запуске возникает ошибка cannot identify image file, убедитесь, что:

Указан правильный путь к Tesseract в .env

Указан путь к Poppler

Загружаемый файл соответствует fileType

pdf2image и pytesseract установлены

📚 Требования (requirements.txt)



📌 TODO (планы на будущее)
 Распознавание текста на английском и других языках

 Выгрузка результата в .docx или .md файл

 Авторизация пользователей

 Логирование и история обработанных файлов

👨‍💻 Автор
Евгений Холмогоров — python backend разработчик, OCR-энтузиаст, любитель FastAPI и чистого кода 😎