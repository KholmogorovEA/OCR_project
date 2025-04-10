from celery import Celery

celery = Celery(
    "tasks",
    broker="redis://localhost:6379",
    include=["app.tasks.tasks", "app.bot.bot"]  
)


from celery.schedules import crontab


celery.conf.beat_schedule = {
    'check-cases-daily': {
        'task': 'app.tasks.tasks.check_all_cases',
        'schedule': crontab(minute="20"),  # Каждый день в 08:20
    },
}
celery.conf.timezone = 'Europe/Moscow' # type: ignore