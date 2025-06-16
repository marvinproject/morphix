from functools import lru_cache
from .app_config import AppConfig

@lru_cache()
def get_app_config() -> AppConfig:
    return AppConfig()
