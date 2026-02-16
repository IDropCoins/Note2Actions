from pathlib import Path
from typing import List, Optional

import zvec

from ingest.core import Chunk


_BACKEND_DIR = Path(__file__).resolve().parent


class ZvecStore:
    def __init__(self, db_path: str):
        self.db_path = str(_BACKEND_DIR / db_path)
        self.collection: Optional[zvec.Collection] = None
        self.vector_field_name = "embedding"

    def _ensure_collection(self, dimension: int) -> None:
        if self.collection is not None:
            return

        path = Path(self.db_path)
        if path.exists():
            self.collection = zvec.open(str(path))
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        schema = zvec.CollectionSchema(
            name="minutesmind_chunks",
            fields=[
                zvec.FieldSchema("doc_id", zvec.DataType.STRING, nullable=False),
                zvec.FieldSchema("source_file", zvec.DataType.STRING, nullable=False),
                zvec.FieldSchema("meeting_date", zvec.DataType.STRING, nullable=False),
                zvec.FieldSchema("chunk_index", zvec.DataType.INT32, nullable=False),
                zvec.FieldSchema("text", zvec.DataType.STRING, nullable=False),
                zvec.FieldSchema("created_at", zvec.DataType.STRING, nullable=False),
                zvec.FieldSchema("tags", zvec.DataType.STRING, nullable=False),
            ],
            vectors=[
                zvec.VectorSchema(
                    self.vector_field_name,
                    zvec.DataType.VECTOR_FP32,
                    dimension=dimension,
                )
            ],
        )
        self.collection = zvec.create_and_open(str(path), schema)

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """
        Insert or update chunk embeddings in Zvec.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Chunk count and vector count mismatch")
        if not chunks:
            return

        self._ensure_collection(len(vectors[0]))
        if self.collection is None:
            raise RuntimeError("Zvec collection was not initialized")

        docs: List[zvec.Doc] = []
        for chunk, vector in zip(chunks, vectors):
            docs.append(
                zvec.Doc(
                    id=chunk.chunk_id,
                    vectors={self.vector_field_name: vector},
                    fields={
                        "doc_id": chunk.doc_id,
                        "source_file": chunk.source_file,
                        "meeting_date": str(chunk.meeting_date),
                        "chunk_index": int(chunk.chunk_index),
                        "text": chunk.text,
                        "created_at": chunk.created_at.isoformat(),
                        "tags": ",".join(chunk.tags),
                    },
                )
            )

        self.collection.upsert(docs)
        self.collection.flush()

    def search(self, query_vector: List[float], top_k: int = 5) -> List[zvec.Doc]:
        """
        Search for nearest chunks by vector similarity.
        """
        if self.collection is None:
            path = Path(self.db_path)
            if path.exists():
                self.collection = zvec.open(str(path))
            else:
                raise RuntimeError("Zvec collection was not initialized")

        return self.collection.query(
            zvec.VectorQuery(field_name=self.vector_field_name, vector=query_vector),
            topk=top_k,
        )