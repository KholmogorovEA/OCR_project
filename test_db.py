from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.sql.model import Case, Subscription  # Импортируйте модели

# Подключение к SQLite
engine = create_engine('sqlite:///./cases.db')
Session = sessionmaker(bind=engine)
db = Session()

# Добавляем тестовое дело
new_case = Case(
    case_number='А40-123123/2028',
    is_active=True,
    state_hash='old_hash',
    last_data='{"State": "Старое состояние"}'
)
db.add(new_case)
db.commit()

# Добавляем подписку на это дело
new_subscription = Subscription(
    user_id=22222,
    case_id=new_case.id  # Используем ID только что добавленного дела
)
db.add(new_subscription)
db.commit()

print("Тестовые данные успешно добавлены.")
db.close()