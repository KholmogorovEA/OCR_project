# update_and_run_celery.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.sql.model import Case  # Импортируйте модель Case

# Подключение к SQLite
engine = create_engine('sqlite:///./cases.db')
Session = sessionmaker(bind=engine)
db = Session()

# Изменение данных в базе
case = db.query(Case).filter(Case.case_number == 'А40-123467/2023').first()  # Находим тестовое дело
if case:
    case.last_data = '{"State": "Новое состояние"}' # type: ignore
    case.state_hash = 'new_hash' # type: ignore
    db.commit()
    print("Данные успешно изменены.")
else:
    print("Дело не найдено.")

db.close()

# Запуск задачи Celery
from app.tasks.tasks import check_all_cases
check_all_cases.delay() # type: ignore
print("Задача Celery запущена.")