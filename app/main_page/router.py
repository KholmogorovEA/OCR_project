import os
import openai
import logging
from typing import List
from pathlib import Path
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, Form, HTTPException, Request, status, APIRouter
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TEMPLATES_FOLDER = "app/templates"

templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

router = APIRouter(
    prefix="/main_page",
    tags=["Главная страница"]
)



# main page
@router.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    logger.debug("Загрузка главной страницы")
    return templates.TemplateResponse("upload.html", {"request": request})