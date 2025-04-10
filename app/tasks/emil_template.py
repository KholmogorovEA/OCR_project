# from email.message import EmailMessage

# from pydantic import EmailStr
# from app.sql.model import Case, Subscription

# from app.config import settings


# def send_to_email(case_data: dict, email_to: EmailStr) -> EmailMessage:
#     """
#     Формирует email с информацией о деле.

#     :param case_data: Данные о деле.
#     :param email_to: Email получателя.
#     :return: Объект EmailMessage.
#     """
#     email = EmailMessage()

#     # Заголовки письма
#     email["Subject"] = f"🔔 Изменения в деле {case_data.get('case_number', 'N/A')}"
#     email["From"] = settings.SMTP_USER
#     email["To"] = email_to

#     # Тело письма
#     email.set_content(
#         f"""
#         <h1>Изменения в деле {case_data.get('case_number', 'N/A')}</h1>
#         <p>Новый статус: {case_data.get('State', 'не указан')}</p>
#         <p>Дата последнего изменения: {case_data.get('last_updated', 'N/A')}</p>
#         """,
#         subtype="html"
#     )

#     return email