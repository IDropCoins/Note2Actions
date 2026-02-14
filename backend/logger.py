import logging

from config import Settings


def setup_logging() -> None:
    settings = Settings()
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
