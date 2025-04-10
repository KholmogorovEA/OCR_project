from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.sql.model import Case, Subscription

# Подключение к SQLite
engine = create_engine('sqlite:///./cases.db')
Session = sessionmaker(bind=engine)
db = Session()



# Проверка данных в таблице Subscription
subscriptions = db.query(Subscription).all()
print("\nПодписки:")
for sub in subscriptions:
    print(f"ID: {sub.id}, User ID: {sub.user_id}, Case ID: {sub.case_id}")

# Проверка данных в таблице Case
cases = db.query(Case).all()
print("\nДела:")
for case in cases:
    print(f"ID: {case.id}, Номер дела: {case.case_number}, Активно: {case.is_active}")

db.close()