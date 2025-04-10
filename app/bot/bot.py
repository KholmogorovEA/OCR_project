import httpx
import http.client
import json
from app.config import settings
from celery import shared_task

TELEGRAM_BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID = settings.TELEGRAM_CHAT_ID


async def send_telegram_message_async(message: str):
    """Запрос к тг боту"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=payload)
        if response.status_code != 200:
            print(f"Ошибка отправки сообщения: {response.text}")


@shared_task
def send_telegram_message(message: str):
    """Запрос к тг боту"""
    conn = http.client.HTTPSConnection("api.telegram.org")
    url = f"/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    })
    
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        conn.request("POST", url, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        
        if response.status == 200:
            print("Сообщение успешно отправлено.")
        else:
            print(f"Ошибка отправки сообщения: {response.status}, {data.decode('utf-8')}")
    except Exception as e:
        print(f"Произошла ошибка при отправке сообщения: {e}")
    finally:
        conn.close()



