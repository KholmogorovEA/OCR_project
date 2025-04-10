import os
import re
import uuid
import openai
import logging
import hashlib
from typing import Dict, Any
from pathlib import Path
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import MarkdownTextSplitter  
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from app.config import settings


load_dotenv()


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) 


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = "app/uploads"
TEMPLATES_FOLDER = "app/templates"
STATIC_FOLDER = "app/static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
URL = settings.URL
model = settings.SENTENCE_TRANSFORMERS_COS_V1
embeddings = HuggingFaceEmbeddings(model_name = model)



def allowed_file(filename: str) -> bool:
    """Функция как валидация переданого файла с front end"""
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'zip', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def filter_data(extractedInformationt: str) -> str: 
    """Функция фильтрации, на тестах вызов llm для исправления орфографии оказался точнее чем либы машинного обучения - DeepPavlov, SpellChecker, и еще 2 libs"""
    messages = [
        {"role": "system", "content": "Ты преподаватель русского языка, твоя задача анализировать текстовые данные после OCR и исправлять орфографические ошибки в словах."},
        {"role": "user", "content": f"Проверь орфографию {extractedInformationt} и верни тот же текст один в один но с исправленными ошибками и БЕЗ ПОВТОРЕНИЯ"}, 
        {"role": "assistant", "content": "Пример: вот слово с ошибкой: побеснокоить, а вот оно же исправленное: побеспокоить"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages, # type: ignore
        temperature=0.1,
        timeout=60   
    )

    response = response.choices[0].message.content
    
    return response # type: ignore



# Пример структуры для few-shot learning
EXAMPLE_METADATA = {
    "document": {
        "title": "Заявление о запрете проведения собрания кредиторов",
        "identifier": {
            "statement_number": "БА-001/1",
            "statement_date": "2022-08-03",
            "file_hash": "0ccc47308fc9f21b073ac4532bf96a16",
            "page_number": 1
        }
    },
    "entities": [
        {
            "type": "organization",
            "name": "ООО 'АРТЕЛЬ'",
            "identifiers": {
                "INN": "5012075458",
                "OGRN": "1125012008395"
            }
        }
    ],
    "financial_data": {
        "accounts": [
            {
                "bank": "ПАО Сбербанк",
                "number": "40702810338000124613",
                "balance": 12045.67,
                "currency": "RUB"
            }
        ]
    }
}

import json

def get_metadata(text: str) -> Dict[str, Any]:
    """Извлекает структурированные метаданные из текста с валидацией и обработкой ошибок."""
    messages = [
        {
            "role": "system",
            "content": (
                "Ты эксперт по извлечению структурированных метаданных. Следуй правилам:\n"
                "1. Используй вложенные структуры\n"
                "2. Форматируй даты как ISO 8601\n"
                "3. Числа как float/int\n"
                "4. Наполняй значения смыслами\n"
                "5. Гарантируй валидный JSON"
            )
        },
        {"role": "user", "content": f"ИЗВЛЕКИ метаданные из текста:\n\n{text}"},
        {"role": "assistant", "content": json.dumps(EXAMPLE_METADATA, ensure_ascii=False, indent=2)}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,  # type: ignore
        temperature=0.2,
        response_format={"type": "json_object"},
        timeout=60
    )
    
    raw_output = response.choices[0].message.content
    logger.debug(f"Raw API response: {raw_output}")

    # Очистка ответа
    raw_output = raw_output.strip()  # type: ignore # Удаляем лишние пробелы
    raw_output = raw_output.lstrip('\ufeff')  # Удаляем BOM

    # Извлечение JSON из ```json ... ```
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, raw_output, re.DOTALL)
    cleaned_output = match.group(1).strip() if match else raw_output

    # Проверка на пустой ответ
    if not cleaned_output:
        logger.warning("Ответ от API пустой или не содержит JSON")
        return {}

    # Десериализация JSON
    try:
        metadata = json.loads(cleaned_output)
        logger.debug(f"Успешно извлечены метаданные: {metadata}")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при разборе JSON: {e}. Ответ: {cleaned_output}")
        return {}



def fetch_and_convert_to_markdown(text: str, name: str) -> str:
    """Функция для преобразования проверенного текста в Markdown разметку"""
    messages = [
        {"role": "system", "content": """Строго следуй следущим просьбе.
        Ты эксперт по форматированию текста и преобразованию его в Markdown.

        Твоя задача — преобразовать предоставленный текст в формат Markdown, используя следующие правила:

        1. Определяй структуру текста и применяй соответствующие заголовки от # до #####, чтобы отразить иерархию содержимого.
        2. Если текст содержит цитаты или важные высказывания, оформляй их с помощью блоков цитирования (>).
        3. Если в тексте есть табличные данные или информация, которая может быть представлена в виде таблицы, создавай таблицы, используя синтаксис Markdown для таблиц.
        4. Сохраняй читаемость и структурированность текста, используя списки, выделение текста (курсив и жирный шрифт), а также другие элементы Markdown там, где это уместно.
        Пример форматирования:

        Заголовки: # Заголовок 1, ## Заголовок 2, ..., ##### Заголовок 5
         
        5. Таблицы:
                
        | Заголовок 1 | Заголовок 2 |
        |-------------|-------------|
        | Значение 1  | Значение 2  |
         
        6. Блоки цитирования: > Цитата или важное высказывание
         
        Текст для преобразования:
        {text}
        Вывод: Возвращай текст, отформатированный в Markdown, следуя указанным правилам.
        """},
        {"role": "user", "content": f"Преобразуй следующий текст в Markdown разметку:\n\n{text}\n используй имя файла:\n\n{name}\n для заголовка Первого уровня. Сохраняй полноту текста"},
        {"role": "assistant", "content": f"""Пример маркдаун разметки:  ## Страница 7: Подробный анализ сделок должника (продолжение)

                                                                        ### Сделка № 2: 
                                                                        - **Контрагент**: Хабибуллина Альбина Анваровна (подозрительная сделка)
                                                                        - **ИНН**: Сведения отсутствуют
                                                                        - **ОГРН**: Не предусмотрено
                                                                        - **Адрес**: Персональные данные
                                                                        - **Договор**:
                                                                        - **Дата**: 23.12.2020
                                                                        - **Сумма**: 182 816,00 руб.
                                                                        - **Номер**: Сведения отсутствуют
                                                                        - **Предмет**: Транспортное средство — Мерседес-Бенц Е200, 2017 г.в.
                                                                            - **Государственный регистрационный знак**: А878РС716
                                                                            - **ММ**: 002130421А263534

                                                                        #### Сведения по сделке:
                                                                        - **Дата**: 23.12.2020
                                                                        - **Сумма**: 182 816,00 руб.
                                                                        - **Передано**: Автомобиль
                                                                        - **Описание**: Продажа автомобиля по цене ниже рыночной стоимости.

                                                                        #### Суждение о сделке:
                                                                        Согласно данным карточки учёта МВД РФ, автомобиль был приобретён должником 08.02.2018 за 2 756 707 руб. и поставлен на временный учёт (лизинг) до 28.02.2021 г. В 2021 году автомобиль был передан Хабибуллиной А.А. за 182 816 руб. (на дату постановки на учёт — 01.04.2021). Рыночная стоимость аналогичных автомобилей начинается с 1 620 000 руб.

                                                                        Сделка может быть признана недействительной в соответствии с п. 1 ст. 61.2 Федерального закона "О несостоятельности (банкротстве)" от 26.10.2002 № 127-ФЗ, если цена сделки значительно отличается от рыночной стоимости. В данном случае автомобиль был продан по цене, значительно ниже рыночной, что указывает на возможное неравноценное встречное исполнение обязательств.

                                                                        Кроме того, сделка может быть признана недействительной, если она была совершена с целью причинить вред имущественным правам кредиторов. Предполагается, что контрагент знал или должен был знать о неплатежеспособности должника и признаках недостаточности его имущества, что делает сделку подозрительной.

                                                                        **Предположительное нарушение**: Сделка совершена на условиях, ухудшающих положение должника, и является оспоримой, поскольку автомобиль был продан в течение одного года до подачи заявления о банкротстве, по цене, существенно ниже рыночной.

                                                                        ---"""}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages,  # type: ignore
        temperature=0.1,
        timeout=60   
    )

    markdown_text = response.choices[0].message.content
    
    return markdown_text  # type: ignore



def get_summary(text: str) -> str:
    """Возьмем краткое описание файлов после OCR для отправки в тг"""
    
    messages = [
        {"role": "system", "content": f"Ты эксперт в обобщении и создании саммари. Сформируй буквально пару (2-3 коротких предложения) кратко, отрази только самую суть в переданном тебе текте \n\n{text}\n"},
        {"role": "user", "content": f"Саммаризируй текст уложись в 150 токенов не больше."}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  
        messages=messages,  # type: ignore
        temperature=0.1,
        max_tokens=150,
        timeout=60
    )

    summary = response.choices[0].message.content

    # Проверяем, если summary пустое или слишком короткое
    if len(summary) == 0: # type: ignore
        logging.warning("Ответ пустой")
        return "Краткое описание невозможно сформировать, много логики и важной информации"
    
    return summary  # type: ignore




def split_and_load(output_file: str) -> list[Document]:
    """Разбивает файл на чанки и создает объекты Document"""
    
    with open(output_file, 'r', encoding='utf-8') as file:
        document_text = file.read()

    # Хэширование содержимого файла
    file_hash = hashlib.md5(document_text.encode('utf-8')).hexdigest()
    logger.info(f"Хэш файла: {file_hash}")

    # Разбивка на чанки
    splitter = MarkdownTextSplitter(chunk_size=2024, chunk_overlap=220)
    chunks = splitter.split_text(document_text)
    logger.info(f"Разбито на {len(chunks)} частей.")

    # Создание документов с метаданными
    documents_chunks = []
    for chunk in chunks:
        metadata = get_metadata(chunk)
        metadata["file_hash"] = file_hash
        doc = Document(page_content=chunk, metadata=metadata)
        doc.id = str(uuid.uuid4())
        documents_chunks.append(doc)
    print("ЧААААААААААААААААААААААААААААННННННННННННННННННННННННННННННННННННККККККККККККККККККККККК/////////////////////////////////////////////////////////////////////", documents_chunks[0])
    return documents_chunks



def create_vector_db(output_file: str, embeddings: Any) -> None:
    """Создает/обновляет векторную базу данных"""
    
    new_documents = split_and_load(output_file)
    
    try:
        vectoreDataBase = FAISS.load_local(
            "app/faiss_soyhanov", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Индекс успешно загружен.")
    except (FileNotFoundError, RuntimeError):
        # Создаем новую базу и завершаем функцию
        logger.info("Создаём новый индекс...")
        vectoreDataBase = FAISS.from_documents(new_documents, embeddings)
        vectoreDataBase.save_local("app/faiss_soyhanov")
        logger.info("Индекс успешно создан.")
        return  # Важно: выход после создания!

    # Фильтрация дубликатов (только для существующей базы)
    existing_hashes = set(
        doc.metadata["file_hash"] 
        for doc in vectoreDataBase.docstore._dict.values()  # type: ignore
        if "file_hash" in doc.metadata
    )
    
    new_unique_documents = [
        doc 
        for doc in new_documents 
        if doc.metadata.get("file_hash") not in existing_hashes
    ]

    # Добавление новых документов
    if new_unique_documents:
        vectoreDataBase.add_documents(new_unique_documents)
        logger.info(f"Добавлено {len(new_unique_documents)} новых документов.")
        vectoreDataBase.save_local("app/faiss_soyhanov")
    else:
        logger.info("Нет новых документов для добавления.")










"""
{
  "document": {
    "title": "Заявление о запрете проведения первого собрания кредиторов ООО 'АРТЕЛЬ'",
    "identifier": {
      "statement_number": "БА-001/1",
      "statement_date": "2022-08-03",
      "file_hash": "0ccc47308fc9f21b073ac4532bf96a16",
      "page_number": 1
    },
    "description": "Документ содержит заявление ООО 'АРТЕЛЬ' о применении обеспечительных мер из-за отсутствия вступивших в законную силу судебных актов по требованиям кредитора Сайханова М.И. Общая сумма требований превышает 1 миллиард рублей."
  },
  "court_case": {
    "case_number": "А41-73921/2020",
    "court_name": "Арбитражный суд Московской области",
    "judge": "Михайлова Наталья Анатольевна",
    "current_status": "Процедура наблюдения",
    "start_date": "2021-08-19",
    "latest_event": {
      "date": "2022-08-01",
      "action": "Отмена апелляционного определения Верховного суда Чеченской Республики и направление дела на новое рассмотрение."
    },
    "relevant_courts": [
      {
        "court_name": "Верховный суд Чеченской Республики",
        "case_number": "33-1117/21",
        "status": "Дело направлено на новое рассмотрение"
      },
      {
        "court_name": "Пятый кассационный суд общей юрисдикции",
        "action": "Отменил апелляционное определение"
      }
    ]
  },
  "creditor": {
    "name": "Сайханов Маратхан Ильманович",
    "claim_details": {
      "amount_rub": 1048891362,
      "status": "Включено в реестр кредиторов, но оспорено в суде",
      "legal_basis": [
        "Решение Гудермесского городского суда Чеченской Республики от 05.04.2021 № 2-88/2021",
        "Апелляционное определение Верховного суда Чеченской Республики от 23.11.2021"
      ]
    },
    "influence": "Сайханов М.И. является крупнейшим кредитором, способным значительно повлиять на результаты голосования собрания кредиторов."
  },
  "company": {
    "name": "ООО 'АРТЕЛЬ'",
    "registration": {
      "OGRN": "1125012008395",
      "INN": "5012075458"
    },
    "address": {
      "full": "143964, Московская область, г. Реутов, ул. Ашхабадская, д. 27, корп. 1, пом. 005, комн. 13",
      "region": "Московская область",
      "city": "Реутов",
      "street": "Ашхабадская",
      "building": "27",
      "room": "13"
    },
    "financial_affiliations": {
      "financial_group": "Финансовая группа ЗЕЛО",
      "beneficiary": "Демчук В.А."
    }
  },
  "temporary_manager": {
    "name": "Аминев Вадим Артурович",
    "contact_info": {
      "region": "Республика Башкортостан",
      "city": "Уфа",
      "PO_box": "33"
    },
    "personal_data": {
      "INN": "027809202648",
      "SNILS": "124-113-402 90"
    },
    "role": "Временный управляющий ООО 'АРТЕЛЬ'",
    "key_actions": [
      {
        "date": "2022-07-20",
        "action": "Опубликовано объявление о созыве собрания кредиторов на 08.08.2022"
      }
    ]
  },
  "events_timeline": [
    {
      "date": "2021-08-19",
      "event": "Введение процедуры наблюдения по делу № А41-73921/2020"
    },
    {
      "date": "2022-06-21",
      "event": "Включение требования Сайханова М.И. в реестр кредиторов"
    },
    {
      "date": "2022-07-20",
      "event": "Публикация объявления о собрании кредиторов"
    },
    {
      "date": "2022-08-01",
      "event": "Отмена апелляционного определения и направление дела на новое рассмотрение"
    }
  ],
  "tags": [
    "банкротство",
    "судебные решения",
    "собрание кредиторов",
    "обеспечительные меры",
    "финансовая группа",
    "временный управляющий"
  ],
  "search_optimization": {
    "keywords": [
      "банкротство ООО 'АРТЕЛЬ'",
      "требование Сайханова",
      "обеспечительные меры",
      "процедура наблюдения",
      "Арбитражный суд Московской области"
    ],
    "summary": "Заявление на запрет собрания кредиторов по делу о банкротстве ООО 'АРТЕЛЬ'. Основной кредитор: Сайханов М.И., сумма требований более 1 млрд рублей. Отсутствуют вступившие в силу судебные акты, кассация направлена на новое рассмотрение."
  }
}

"""