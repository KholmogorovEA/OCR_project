from sqlalchemy import Column, String, JSON, Boolean, Integer, ForeignKey
from app.database import Base


class Case(Base):
    __tablename__ = "cases"
    
    id = Column(Integer, primary_key=True)
    case_number = Column(String, unique=True)  # Номер дела (уникальный)
    last_data = Column(JSON)  # Последние полученные данные
    state_hash = Column(String)  # Хэш состояния для быстрого сравнения
    is_active = Column(Boolean, default=True)  # Флаг активности отслеживания
    

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)  # ID пользователя в Telegram
    case_id = Column(Integer, ForeignKey('cases.id'))
    
 