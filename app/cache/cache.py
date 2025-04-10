from functools import lru_cache
from fastapi.templating import Jinja2Templates

TEMPLATES_FOLDER = "app/templates"
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)

@lru_cache(maxsize=20)
def render_cached_template(template_name: str, context: dict):
    return templates.TemplateResponse(template_name, context)