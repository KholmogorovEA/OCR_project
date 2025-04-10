import os
import re
import openai
import zipfile
import logging
import pytesseract
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from typing import List, Dict, Any
from pydantic import BaseModel
from pathlib import Path
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
import faiss


from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


from fastapi import FastAPI, UploadFile, Form, HTTPException, Request, status, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from app.config import settings
from app.tools.funcs import allowed_file, filter_data, create_vector_db, fetch_and_convert_to_markdown, get_summary, split_and_load

from app.bot.bot import send_telegram_message
from app.database import SessionLocal, engine
from app.sql.model import Base, Case, Subscription
from sqlalchemy.orm import Session
from app.tasks.tasks import *
from app.upload_data.router import router as router_upload
from app.main_page.router import router as router_main_page
# from app.chat.router import router as router_chat
# from app.cache.cache import render_cached_template # type: ignore
from langchain.chains.router import MultiRetrievalQAChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from vector_db_st_from_hf import MarkdownProcessor
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_completion_tokens=6000, # type: ignore
                timeout=30,
                max_retries=2,
                api_key=settings.OPENAI_API_KEY, # type: ignore
            )



load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # чекнуть как при деплои быть!!!!!!!!!!!!!!!!!!!!
poppler_path = r"C:\Program Files (x86)\poppler-24.08.0\Library\bin"                     # чекнуть как при деплои быть!!!!!!!!!!!!!!!!!!!!

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) 


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = "app/uploads"
TEMPLATES_FOLDER = "app/templates"
STATIC_FOLDER = "app/static"
SCRIPTS_FOLDER = BASE_DIR / "scripts"
UPLOAD_FOLDER = BASE_DIR / "app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
URL = settings.URL
model_V1 = settings.SENTENCE_TRANSFORMERS_COS_V1
embeddings_V1 = HuggingFaceEmbeddings(model_name = model_V1)
# model_v2 = settings.SENTENCE_TRANSFORMERS_BASE_V2
# embeddings_V2 = HuggingFaceEmbeddings(model_name = model_v2)
db = FAISS.load_local("app/faiss_dialog_invest", embeddings_V1, allow_dangerous_deserialization=True) # type: ignore
# faiss_dialog_invest_cos_v1 = FAISS.load_local("app/faiss_dialog_invest", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})
# faiss_other_data_cos_v1 = FAISS.load_local("app/faiss_other_data", embeddings_V1, allow_dangerous_deserialization=True)
# faiss_sedoy_cos_v1 = FAISS.load_local("app/faiss_sedoy", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})
# faiss_zakaryan_ashot_cos_v1 = FAISS.load_local("app/faiss_zakaryan_ashot", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})
# faiss_zakaryan_susanna_cos_v1 = FAISS.load_local("app/faiss_zakaryan_susanna", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})
# faiss_new_documents_cos_v1 = FAISS.load_local("app/faiss_new_documents", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})
# faiss_soyhanov_cos_v1 = FAISS.load_local("app/faiss_soyhanov", embeddings_V1, allow_dangerous_deserialization=True).as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.55})

# index_deep_pavlov_processed_text = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\processed_text_faiss_index.index") # type: ignore
# index_deep_pavlov_dialog_invest = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\dialog_invest_faiss_index.index") # type: ignore
# index_deep_pavlov_sedoy = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\sedoy_faiss_index.index") # type: ignore
# index_deep_pavlov_zakaryan_ashot = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\zakaryan_ashot_faiss_index.index") # type: ignore
# index_deep_pavlov_zakaryan_susanna = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\transformersDeepPavlov\\zakaryan_susanna_faiss_index.index") # type: ignore

# index_sentence_transformers_processed_text = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\processed_text_faiss_index.index") # type: ignore
# index_sentence_transformers_dialog_invest = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\dialog_invest_faiss_index.index") # type: ignore
# index_sentence_transformers_sedoy = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\sedoy_faiss_index.index") # type: ignore
# index_sentence_transformers_zakaryan_ashot = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\zakaryan_ashot_faiss_index.index") # type: ignore
# index_sentence_transformers_zakaryan_susanna = faiss.read_index("C:\\Users\\Evgenii\\Desktop\\OCR_bot\\sent_transformers\\zakaryan_susanna_faiss_index.index") # type: ignore

# MD_FOLDER = "markdown_docs"
# processor = MarkdownProcessor()
# all_documents_dialog_invest = processor.process_markdown_files(MD_FOLDER)
# print(f"✅ Создано {len(all_documents_dialog_invest)} чанков")

# retriever_index_sentence_transformers_processed_text = FAISS(embedding_function=embeddings_V2, index=index_sentence_transformers_processed_text).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.55}) # type: ignore
# retriever_index_sentence_transformers_dialog_invest = FAISS(all_documents_dialog_invest ,embedding_function=embeddings_V2).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.55}) # type: ignore
# retriever_index_sentence_transformers_sedoy = FAISS(embedding_function=embeddings_V2, index=index_sentence_transformers_sedoy).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.55}) # type: ignore
# retriever_index_sentence_transformers_zakaryan_ashot = FAISS(embedding_function=embeddings_V2, index=index_sentence_transformers_zakaryan_ashot).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.55}) # type: ignore
# retriever_index_sentence_transformers_zakaryan_susanna = FAISS(embedding_function=embeddings_V2, index=index_sentence_transformers_zakaryan_susanna).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.55}) # type: ignore

















def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def is_pdf(filename):
    return filename.lower().endswith('.pdf')

def is_zip(filename):
    return filename.lower().endswith('.zip')







# Conn to the database
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="OCR Webservice для команды юристов", 
    description="Проект переводит сканированные документы в текст, и находит соответсвующую информацию в чате с нейро-юристом",
    version="1.0.0"
    )
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# for swagger
app.include_router(router_upload) 
app.include_router(router_main_page)
# app.include_router(router_chat)


origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8000/chat",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin", "Authorization"],)


# conn to Redis
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    redis = aioredis.from_url("redis://localhost:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="cache")
    yield



# main page
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    logger.debug("Загрузка главной страницы")
    return templates.TemplateResponse("upload.html", {"request": request})



def truncate_text(texts: List[str], max_tokens: int) -> List[str]:
    """Обрезаем текст, если он превышает лимит токенов"""
    total_tokens = count_tokens(texts)
    if total_tokens > max_tokens:
        avg_tokens_per_text = total_tokens // len(texts)
        max_texts = max_tokens // avg_tokens_per_text
        return texts[:max_texts]  # Обрезаем до допустимого числа элементов
    return texts




import tiktoken

def count_tokens(texts: List[str]) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    total_text = " ".join(texts)
    return len(enc.encode(total_text))



# Обработка OCR этапы 2,3,4 из Тех задания
@app.post("/")
def upload_file(file: UploadFile, fileType: str = Form(...)):
    doc_type = Path(file.filename).stem
    filename = UPLOAD_FOLDER / file.filename
    with open(filename, "wb") as f:
        f.write(file.file.read())

    extracted_texts = []

    if fileType == "single" and is_image(file.filename):
        image = Image.open(filename)
        text = pytesseract.image_to_string(image, lang='rus')
        corrected_text = filter_data(text)
        markdawn = fetch_and_convert_to_markdown(corrected_text, doc_type)
        extracted_texts.append(markdawn)

    elif fileType == "archive" and is_zip(file.filename):
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)
                for extracted_file in zip_ref.namelist():
                    if is_image(extracted_file):
                        extracted_path = UPLOAD_FOLDER / extracted_file
                        with open(extracted_path, 'rb') as img_file:
                            image = Image.open(img_file)
                            text = pytesseract.image_to_string(image, lang='rus')
                            corrected_text = filter_data(text)
                            markdawn = fetch_and_convert_to_markdown(corrected_text, doc_type)
                            extracted_texts.append(markdawn)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Невалидный ZIP-файл")

    elif fileType == "pdf" and is_pdf(file.filename):
        if os.path.exists(filename):
            images = convert_from_path(filename, 300, poppler_path=poppler_path)
            for page_number, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='rus')
                markdawn = fetch_and_convert_to_markdown(page_text, doc_type)
                extracted_texts.append(f"Страница {page_number + 1}:\n{markdawn}\n")
        else:
            raise HTTPException(status_code=404, detail="Файл не найден")

    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла")

    # Сохраняем текст в файл
    output_file = UPLOAD_FOLDER / "soyhanov.md"  # type: ignore
    with open(output_file, 'a', encoding='utf-8') as out_f:
        out_f.write("\n".join(extracted_texts) + "\n\n НОВАЯ СТРОООООООООООООООООООКККААААААААА" + 3 * "--------------------------------------------------------")  # type: ignore

    
    # Обновляем векторную базу данных
    create_vector_db(output_file, embeddings_V1)  # type: ignore


    # Проверем длину extracted_texts
    token_count = count_tokens(extracted_texts)
    MAX_TOKENS = 14000

    if token_count > MAX_TOKENS:
        summary = "Файл слишком большой для обработки. Суммаризация пропущена."
        send_telegram_message(f"Новый документ добавлен в векторную базу: {doc_type}\n")
        send_telegram_message(f"Файл слишком большой для обработки. Количество токенов: {token_count} (максимум {MAX_TOKENS}).\n")
    else:
        extracted_texts = truncate_text(extracted_texts, MAX_TOKENS)
        summary = get_summary(extracted_texts)  # type: ignore
        send_telegram_message(f"Новый документ добавлен в векторную базу: {doc_type}\n")
        send_telegram_message(f"Краткое описание: {summary}\n")

    # Читаем весь текст для рендера
    with open(output_file, 'r', encoding='utf-8') as in_f:
        all_texts = in_f.read()

    # Рендер
    return templates.TemplateResponse("rezult.html", {"request": {}, "processed_texts": all_texts})



# Валидируем данные с frontenda
class MessageRequest(BaseModel):
    topic: str
    system_for_NA: str
    
    


# # ██ Конфигурация ретрейверов
# retriever_infos = [
#     {
#         "name": "Банкротсво Диалог Инвест",
#         "description": """Suitable for answering questions about Application of ZELO LLC for inclusion in the register of claims of Dialog Invest with an inventory and a check. Application of OOO ZELO on reduction of the amount of claims against the debtor on 3l. Debt calculation from the Tverskoy Court case. Notice of the meeting of creditors of Dialog Invest on 22.08.2023. 2024-10-14 Notice of the meeting of creditors of Dialog Invest (inc.18.10.24) State duty on appeal Zelo. A32-28626-2021_20221024_Reshenija_i_postanovlenija Признание банкротом. A32-28626-2021_20230619_Opredelenie On the Election of the Arbitration Administrator. A32-28626-2021_20231113_Opredelenie Inclusion of Zelo Claims. A32-28626-2021_20240603_Opredelenie Postponement of the procedure. A32-28626-2021_20240816_Opredelenie Remanding the succession to Vladimirska without motion. A32-28626-2021_20240816_Opredelenie Termination of the procedure. A32-28626-2021_20241121_Postanovlenie_apelljacionnoj_instancii Отмена прекращения.""",
#         "examples": ["Заявление ООО ЗЕЛО о включении в реестр требований Диалог Инвест с описью и чеком", "Заявление ООО ЗЕЛО об уменьшении суммы требований к должнику на 3л", "Уведомление о собрании кредиторов Диалог Инвест", "A32-28626-2021_20221024_Reshenija_i_postanovlenija Признание банкротом", "A32-28626-2021_20240816_Opredelenie Прекращение процедуры"],
#         "retriever": faiss_dialog_invest_cos_v1,  # Заменить на реальный ретрейвер
#     },
#     {
#         "name": "Банкротсво Седого",
#         "description": """Suitable for answering questions about Appeal in case 2-523-2022 from Sedoy A.A. (inh.19.03.24). 2024-04-15 Appeal. complaint from Sedoy A.A. case A32-48721_2023 (inh.26.04.24) 2024-04-15 Appeal. complaint from Sedoy A.A. case A32-48721_2023 (inh.26.04.24)-1. Notification about the meeting of creditors by Sedoy A.A. (inh.21.05.24). A32-48721-2023_20231009_Opredelenie Acceptance of the application. A32-48721-2023_20231206_Opredelenie. A32-48721-2023_20240227_Opredelenie Введение реструктуризации. A32-48721-2023_20240517_Opredelenie. A32-48721-2023_20240909_Opredelenie_Протокол. Sedoy is address #1. Sedoy A.A. to address #2.""",
#         "examples": ["Апелляционная жалоба по делу 2-523-2022 от Седой А.А. (вх.19.03.24)", "Уведомление о собрании кредиторов Седой А.А. (вх.21.05.24)", "A32-48721-2023_20240227_Opredelenie Введение реструктуризации", "Седой - адрес №1. Седой - адрес №2."],
#         "retriever": faiss_sedoy_cos_v1,
#     },
#     {
#         "name": "Банкротство Закарян Ашота",
#         "description": """Suitable for answering questions about ZELO LLC bankruptcy application of ZAKARYAN A.A. Все про Закаряна Ашота. with an inventory and a check. Notice of the meeting of creditors Zakaryan A.A. (vh.21.05.24) A41-35102-2023_20230607_Opredelenie. A41-35102-2023_20241107_Opredelenie Replacement of Zelo by Vladimirskaya in part of the claims. Zorya to Zakarian for 26 mln. Kozeruk to Zakarian under the receipt. Decision on the claim of MC Veles Management to Zakaryans. A41-35102-2023_20231207_Opredelenie On reclamation of evidence. A41-35102-2023_20231207_Opredelenie On refusal to demand evidence from the Civil Registry Office. A41-35102-2023_20240326_Opredelenie On transfer of documents. A41-35102-2023_20241010_Opredelenie Assignment of claims to Artel by Subsidiary""",
#         "examples": ["очередь наследства", "завещание"],
#         "retriever": faiss_zakaryan_ashot_cos_v1,
#     },
#     {
#         "name": "Банкротство Закарян Сюзанна",
#         "description": "Suitable for answering questions about Suzanne Zakarian and all information related to her. Kopija dogovora zajma 27012020015-MSB ot 27 janvarja 2020 goda. 4.Kopija dogovora poruchitelstva ot 27 janvarja 2020 goda. Kopija dogovoru ustupki 04-03-22 BD-Faktorius ot 04 marta 2022 goda. AO_PKO_YUB_FAKTORIUS. ZELO bankruptcy petition of Zakaryan Susanna with inventory and check. Opredelenie Acceptance of the application.Opredelenie Acceptance of the application of Zelo (Susanna). A41-30605-2023_20230706_Opredelenie. Leaving the Factorius application without consideration. Reshenija_i_postanovlenija Recognition of Susanna as bankrupt. Claims of SME Bank. Opredelenie Claims of Artel without motion.",
#         "examples": ["Закарян Сюзанна", "Заявление ЗЕЛО о банкротстве Закарян Сусанны с описью и чеком", "ставление заявления Факториус без рассмотрения", "Reshenija_i_postanovlenija Признание Сусанны банкротом"],
#         "retriever": faiss_zakaryan_susanna_cos_v1,
#     },
#     {
#         "name": "Банкротство Артель",
#         "description": "Suitable for answering questions about bankruptcy ARTEL. ARTEL - APPLICATION (for an injunction against a PSC). Power of attorney notarized by Artel for Zakaryan Ashot. Statement of accounts of Artel. Power of Attorney Artel for Altshuler. Analysis of financial condition of LLC ARTEL. Conclusion on transactions Finanalysis of LLCARTEL. answers of reg bodies of the Ministry of Internal Affairs (1). answers of reg bodies. Report on the activities of the temporary manager in the AC dated 20.07.2022. Submission of the candidacy of the insolvency administrator 1l. Minutes of the meeting of creditors 2. Register of creditors as of 29.07.2022. Claims behind the register of creditors_ as of 29.07.2022. Notifications and requests.",
#         "examples": ["банкротство Артель", "собрание кредиторов", "Доверенность нотариальная Артель на Закаряня Ашота", "редставление кандидатуры арбитражного управляющего 1л. Протокол собрания кредиторов 2. Реестр кредиторов на 29.07.2022. Требования за реестром кредиторов_на 29.07.2022. Уведомления и запросы"],
#         "retriever": faiss_other_data_cos_v1,
#     },
# ]


# chain = MultiRetrievalQAChain.from_retrievers(
#     llm=llm,
#     retriever_infos=retriever_infos,
#     default_chain_llm=llm,  # Передаём default_chain_llm, чтобы не получать ошибку
#     verbose=True
# )

# chain.run("")


#Роут для получения сообщений frontend -> backend -> llm -> frontend
@app.post("/chat")
async def chat_with_lawyer(request: MessageRequest) -> dict[str, str]:
    """Чат с llm по вектоной базе"""
    try:
        docs = db.max_marginal_relevance_search(query=request.topic, k=10, fetch_k=75, lambda_mult=0.8)  # type: ignore
        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

        messages = [
            {"role": "system", "content": request.system_for_NA},
            {"role": "user", "content": f"Ответь на вопрос пользователя. Не упоминай отрывки документов с информацией для ответа. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{request.topic}"}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            temperature=0,
            timeout=60,
            max_completion_tokens=1500
        )


        # Удаление звездочек и Markdown разметки
        raw_answer = completion.choices[0].message.content.strip()  # type: ignore
        clean_answer = re.sub(r'[\*\_#>`]', '', raw_answer).replace('**', '').replace('__', '').strip()

        # Логирование ответа перед возвратом
        print("Ответ от LLM:", clean_answer)
        return {"answer": clean_answer}
    except Exception as e:
        print("Ошибка:", str(e))
        raise HTTPException(status_code=500, detail=str(e))







# ========================================================================REST FULL API KAD.ARBITR.RU============================================================================




# Роут: tg -> here -> db -> celery1 -> celery2 -> tg
@app.post("/tracking")
async def track_case_endpoint(case_number: str, user_id: int, db: Session = Depends(get_db)) -> dict[str, str]: # type: ignore
    """ Добавление дела в базу и отслеживает дела, запускает задачу Celery """
    case = db.query(Case).filter(Case.case_number == case_number).first()
    
    if not case:
        # Создаем новое дело
        new_case = Case(
            case_number=case_number,
            last_data={},
            state_hash="",
            is_active=True
        )
        db.add(new_case)
        db.commit()
        db.refresh(new_case)
        case = new_case


    # Добавляем подписку
    subscription = db.query(Subscription).filter(
        Subscription.user_id == user_id,
        Subscription.case_id == case.id
    ).first()
    
    if not subscription:
        new_sub = Subscription(user_id=user_id, case_id=case.id)
        db.add(new_sub)
        db.commit()

    # Запускаем задачу Celery
    track_case.delay(case_number) # type: ignore
    
    return {"status": "tracking_started"}



# Роут отписки от дел которые завершились 
@app.post("/unsubscribe")
async def unsubscribe_case(case_number: str, user_id: int, db: Session = Depends(get_db)): # type: ignore
    """
    Удаляет подписку пользователя на дело.
    Если у дела больше нет подписчиков, оно помечается как неактивное.
    """
    try:
        # Ищем дело в базе данных
        case = db.query(Case).filter(Case.case_number == case_number).first()
        if not case:
            raise HTTPException(status_code=404, detail="Дело не найдено.")
        
        # Ищем подписку пользователя
        subscription = db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.case_id == case.id
        ).first()
        
        if not subscription:
            return {"message": "Вы не отслеживаете это дело."}
        
        # Удаляем подписку
        db.delete(subscription)
        db.commit()
        
        # Проверяем, остались ли подписчики у дела
        remaining_subscriptions = db.query(Subscription).filter(
            Subscription.case_id == case.id
        ).count()
        
        if remaining_subscriptions == 0:
            # Если подписчиков нет, помечаем дело как неактивное
            case.is_active = False
            db.commit()
        
        return {"message": f"Вы отписались от дела {case_number}."}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



# Роут получения списка всех дел 
@app.get("/cases/summary")
async def get_cases_summary(user_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]: # type: ignore
    """
    Возвращает список дел пользователя с их номерами и статусами.

    :param user_id: ID пользователя.
    :param db: Сессия базы данных.
    :return: Список дел с номерами и статусами.
    """
    try:
        # Получаем все подписки пользователя
        subscriptions = db.query(Subscription).filter(Subscription.user_id == user_id).all()
        
        if not subscriptions:
            return {"cases": []}  # Возвращаем пустой список, если подписок нет
        
        # Получаем данные о делах
        cases_summary: List[Dict[str, Any]] = []
        for sub in subscriptions:
            case = db.query(Case).filter(Case.id == sub.case_id).first()
            if case:
                cases_summary.append({
                    "case_number": case.case_number,
                    "status": case.last_data.get("State", "не указан")
                })
        
        return {"cases": cases_summary}
    
    except Exception as e:
        # Логируем ошибку
        logger.error(f"Ошибка при получении списка дел для пользователя {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Произошла внутренняя ошибка сервера.")





app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
