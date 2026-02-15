import argparse
import logging
import os

from config import Settings
from ingest.core import chunk_document, discover_note_files, infer_meeting_date, load_text
from ingest.embedding import Embedder
from logger import setup_logging
from storage import ZvecStore

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
    logger.info("Zvec path: %s", settings.zvec_db_path)
    logger.info("Embedding model: %s", settings.embed_model)

    files = discover_note_files(args.path)
    logger.info("Discovered %d note files", len(files))
    embedder = Embedder(settings.embed_model, batch_size=args.batch_size)
    store = ZvecStore(settings.zvec_db_path)

    for abs_path, rel_path in files:
        stat = os.stat(abs_path)
        text = load_text(abs_path)
        meeting_date = infer_meeting_date(
            relative_path=rel_path,
            text=text,
            file_mtime=stat.st_mtime,
        )
        doc_id, chunks = chunk_document(
            source_file=rel_path,
            text=text,
            meeting_date=meeting_date,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )

        texts = [chunk.text for chunk in chunks]
        vectors = embedder.embed(texts) if texts else []
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch for {rel_path}: {len(vectors)} != {len(chunks)}"
            )

        store.upsert(chunks, vectors)

        logger.info("Meeting date for %s -> %s", rel_path, meeting_date)
        logger.info("Document ID for %s -> %s", rel_path, doc_id)
        logger.info("Created %d chunks for %s", len(chunks), rel_path)
        logger.info("Generated %d embeddings for %s", len(vectors), rel_path)
        logger.info("Stored %d chunks in Zvec for %s", len(chunks), rel_path)
        logger.info("Loaded %s (%d chars)", rel_path, len(text))


if __name__ == "__main__":
    main()
