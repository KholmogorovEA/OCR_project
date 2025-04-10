import httpx
import hashlib
import json
import smtplib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.tasks.celery import celery
from app.sql.model import Case, Subscription
from app.database import sessionmaker
from app.bot.bot import send_telegram_message
# from app.tasks.emil_template import send_to_email



engine = create_engine("sqlite:///./cases.db")    # env
Session = sessionmaker(bind=engine)


def get_case_data(case_number: str):
    """
    Парсинг данных с kad.arbitr.ru
    """
    try:
        url = "https://parser-api.com/parser/arbitr_api/run.php"  # URL API
        params = {
            "key": "ccf3ce57f02f0326399fadda323ac14b",  # API ключ
            "CaseNumber": case_number
        }

        # Отправляем запрос
        response = httpx.get(url, params=params)
        response.raise_for_status()

        # Преобразуем ответ в JSON
        data = response.json()

        # Выводим в консоль в удобном виде
        print("\n--- Полученные данные дела ---")
        print(f"Номер дела: {case_number}")
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        print("\n--- Конец данных ---")

        return data

    except httpx.HTTPStatusError as e:
        print(f"HTTP ошибка: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



@celery.task
def track_case(case_number: str):
    db = Session()
    try:
        data = get_case_data(case_number)
        state_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        case = db.query(Case).filter(Case.case_number == case_number).first() # type: ignore
        case.last_data = data # type: ignore
        case.state_hash = state_hash # type: ignore
        db.commit()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        db.close()



@celery.task
def check_all_cases():
    db = Session()
    try:
        cases = db.query(Case).filter(Case.is_active == True).all()
        for case in cases:
            current_hash = case.state_hash
            new_data = get_case_data(case.case_number)  # type: ignore # Получаем новые данные
            new_hash = hashlib.md5(json.dumps(new_data, sort_keys=True).encode()).hexdigest()
            
            if new_hash != current_hash:
                print(f"Хэш изменился для дела {case.case_number}. Отправляем уведомления.")
                # Обновляем данные
                case.last_data = new_data # type: ignore
                case.state_hash = new_hash # type: ignore
                db.commit()
                
                # Отправляем уведомления подписчикам
                subscriptions = db.query(Subscription).filter(Subscription.case_id == case.id).all()
                for sub in subscriptions:
                    print(sub)
                    send_telegram_message.delay( # type: ignore
                        message=f"🔔 Изменения в деле {case.case_number}!\n"
                               f"Новый статус: {new_data.get('State', 'не указан')}" # type: ignore
                    )
                    # emails_to_notify = ["demchukrus@gmail.com", "bykholmogoro@gmail.com"]
                    # for email in emails_to_notify:
                    #     send_confirmation_case.delay(email, new_data) # type: ignore
            else:
                print(f"Хэш не изменился для дела {case.case_number}. Уведомления не требуются.")
    except Exception as e:
        print(f"Ошибка при проверке дел: {str(e)}")
    finally:
        db.close()
            
  

# @celery.task
# def send_confirmation_case(email_to: EmailStr, case_data: dict):
#     """
#     Отправляет email с подтверждением изменений в деле.

#     :param email_to: Email получателя.
#     :param case_data: Данные о деле.
#     """
#     try:
#         # Формируем письмо
#         msg_content = send_to_email(case_data, email_to)

#         # Отправляем письмо через SMTP
#         with smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT) as server:
#             server.login(settings.SMTP_USER, settings.SMTP_PASS)
#             server.send_message(msg_content)
#             print(f"Email confirmation sent to {email_to}.")
#     except Exception as e:
#         print(f"Ошибка при отправке email на {email_to}: {str(e)}")
    