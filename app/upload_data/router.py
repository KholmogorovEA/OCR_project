import os
import zipfile
import pytesseract
from pdf2image import convert_from_path
from typing import List
from pathlib import Path
from fastapi import APIRouter, FastAPI, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from dotenv import load_dotenv
from app.config import settings
from app.tools.funcs import allowed_file, filter_data, create_vector_db, fetch_and_convert_to_markdown, get_summary
from app.bot.bot import send_telegram_message


load_dotenv()
poppler_path = r"C:\Program Files (x86)\poppler-24.08.0\Library\bin" 

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = "app/uploads"
TEMPLATES_FOLDER = "app/templates"
STATIC_FOLDER = "app/static"
SCRIPTS_FOLDER = BASE_DIR / "scripts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
URL = settings.URL
model = settings.SENTENCE_TRANSFORMERS_COS_V1
embeddings = HuggingFaceEmbeddings(model_name = model)
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)



router = APIRouter(
    prefix="/upload_data",
    tags=["Загрузка сканов/pdf/zip архива для извлечения текстовой информации при помощи предобученного Pytesseract OCR "]
)



@router.post("/")
def upload_file(file: UploadFile, fileType: str = Form(...)):

    doc_type = Path(file.filename).stem # type: ignore
    filename = UPLOAD_FOLDER / file.filename # type: ignore
    with open(filename, "wb") as f:
         f.write(file.file.read())

    extracted_texts = []

    if fileType == "single" and allowed_file(file.filename): # type: ignore
        image = Image.open(filename)
        text = pytesseract.image_to_string(image, lang='rus')
        corrected_text = filter_data(text)
        markdawn = fetch_and_convert_to_markdown(corrected_text, doc_type)
        extracted_texts.append(markdawn)

    elif fileType == "archive" and allowed_file(file.filename): # type: ignore
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
                for extracted_file in zip_ref.namelist():
                    if allowed_file(extracted_file):
                        extracted_path = UPLOAD_FOLDER / extracted_file # type: ignore
                        with open(extracted_path, 'rb') as img_file:
                            image = Image.open(img_file)
                            text = pytesseract.image_to_string(image, lang='rus')
                            corrected_text = filter_data(text)
                            markdawn = fetch_and_convert_to_markdown(corrected_text, doc_type)
                            extracted_texts.append(markdawn)

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Невалидный ZIP-файл")

    elif fileType == "pdf" and allowed_file(file.filename): # type: ignore
        if os.path.exists(filename):
            images = convert_from_path(filename, 300, poppler_path=poppler_path)
            for page_number, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='rus')
                corrected_text = filter_data(page_text)
                markdawn = fetch_and_convert_to_markdown(corrected_text, doc_type)
                extracted_texts.append(f"Страница {page_number + 1}:\n{markdawn}\n")
        else:
            raise HTTPException(status_code=404, detail="Файл не найден")

    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла")
    
    #саммари
    summary = get_summary(extracted_texts)  # type: ignore

    # сохраняем а затем и далее добавляем текст в файл
    output_file = UPLOAD_FOLDER / "processed_text.txt" # type: ignore
    with open(output_file, 'a', encoding='utf-8') as out_f:
        out_f.write("\n".join(extracted_texts) + "\n\n НОВАЯ СТРОООООООООООООООООООКККААААААААА" + 3*"--------------------------------------------------------") # type: ignore

    # db
    create_vector_db(output_file, embeddings) # type: ignore

    send_telegram_message(f"Новый документ добавлен в векторную базу: {doc_type}\n") # type: ignore
    send_telegram_message(f"Краткое описание: {summary}\n") # type: ignore

    # читаем весь текст для рендера
    with open(output_file, 'r', encoding='utf-8') as in_f:
        all_texts = in_f.read()

    # рендер
    return templates.TemplateResponse("rezult.html", {"request": {}, "processed_texts": all_texts})