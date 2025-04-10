from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    SENTENCE_TRANSFORMERS_COS_V1: str
    SENTENCE_TRANSFORMERS_BASE_V2: str
    OPENAI_API_KEY: str
    URL: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    USER_AGENT: str
    # SMTP_HOST: str
    # SMTP_PORT: int
    # SMTP_USER: str
    # SMTP_PASS: str

    # REDIS_HOST: str
    # REDIS_PORT: int


    class Config:
        env_file = ".env"

settings = Settings()  # type: ignore
