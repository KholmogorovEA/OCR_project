# import os
# import re
# import openai
# import logging
# import requests
# import faiss
# from typing import List
# from pathlib import Path
# from fastapi import APIRouter, HTTPException
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.templating import Jinja2Templates
# from langchain_community.vectorstores import FAISS 
# from langchain_huggingface import HuggingFaceEmbeddings
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from app.config import settings


# TEMPLATES_FOLDER = "app/templates"
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
# client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) 
# model = settings.SENTENCE_TRANSFORMERS
# embeddings = HuggingFaceEmbeddings(model_name = model)
# index = FAISS.load_local("app/faiss_bankruptcy_artel", embeddings, allow_dangerous_deserialization=True)


# templates = Jinja2Templates(directory=TEMPLATES_FOLDER)


# router = APIRouter(
#     prefix="/chat",
#     tags=["Чат с Нейро-юристом"]
# )


# class MessageRequest(BaseModel):
#     topic: str
#     system_for_NA: str
    
    
# #Роут для получения сообщений и отправки их в LLM
# @router.post("/chat")
# async def chat_with_lawyer(request: MessageRequest) -> dict[str, str]:
#     try:
#         docs = index.max_marginal_relevance_search(query=request.topic, k=10, fetch_k=75, lambda_mult=0.4) 
#         message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, doc in enumerate(docs)]))

#         messages = [
#             {"role": "system", "content": request.system_for_NA},
#             {"role": "user", "content": f"Ответь на вопрос пользователя. Не упоминай отрывки документов с информацией для ответа. Документ с информацией для ответа: {message_content}\n\nВопрос пользователя: \n{request.topic}"}
#         ]

#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages, # type: ignore
#             temperature=0,
#             timeout=60
#         )

#         # Логирование ответа перед возвратом
#         print("Ответ от LLM:", completion.choices[0].message.content)
#         return {"answer": completion.choices[0].message.content.strip()} # type: ignore
#     except Exception as e:
#         print("Ошибка:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))
