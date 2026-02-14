import argparse
import logging

from config import Settings
from discover import discover_note_files
from loader import load_text
from logger import setup_logging


logger = logging.getLogger("minutesmind.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="MinutesMind ingestion CLI")
    parser.add_argument("--path", required=True, help="Folder containing .md/.txt notes")
    parser.add_argument("--rebuild", action="store_true", help="Reingest all files")
    parser.add_argument("--batch-size", type=int, default=48, help="Embedding batch size")

    args = parser.parse_args()

    setup_logging()
    settings = Settings()

    logger.info("CLI initialized")
    logger.info("Path: %s", args.path)
    logger.info("Rebuild: %s", args.rebuild)
    logger.info("Batch size: %s", args.batch_size)
    logger.info("Zvec path: %s", settings.ZVEC_DB_PATH)
    logger.info("Embedding model: %s", settings.EMBED_MODEL)

    files = discover_note_files(args.path)
    logger.info("Discovered %d note files", len(files))

    for abs_path, rel_path in files:
        text = load_text(abs_path)
        logger.info("Loaded %s (%d chars)", rel_path, len(text))


if __name__ == "__main__":
    main()
