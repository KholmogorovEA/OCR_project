

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



from app.tools.funcs import get_metadata










import os

output_file = "C:\\Users\\Evgenii\\Desktop\\OCR_bot\\app\\app\\uploads\\zakaryan_susanna.txt"

def split_and_load(output_file: str) -> list[Document]:
    """Разбивает файл на чанки и создает объекты Document"""
    
    with open(output_file, 'r', encoding='utf-8') as file:
        document_text = file.read()

    # Хэширование содержимого файла
    file_hash = hashlib.md5(document_text.encode('utf-8')).hexdigest()
    logger.info(f"Хэш файла: {file_hash}")

    # Разбивка на чанки
    splitter = MarkdownTextSplitter(chunk_size=2500, chunk_overlap=250)
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
            "app/faiss_zakaryan_susanna", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Индекс успешно загружен.")
    except (FileNotFoundError, RuntimeError):
        # Создаем новую базу и завершаем функцию
        logger.info("Создаём новый индекс...")
        vectoreDataBase = FAISS.from_documents(new_documents, embeddings)
        vectoreDataBase.save_local("app/faiss_zakaryan_susanna")
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
        vectoreDataBase.save_local("app/faiss_zakaryan_susanna")
    else:
        logger.info("Нет новых документов для добавления.")


create_vector_db(output_file, embeddings)