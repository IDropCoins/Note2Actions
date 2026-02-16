"""Structured extraction from meeting notes using LangExtract."""

from typing import Any, Dict, List

import langextract as lx
from langextract.data import CharInterval, ExampleData, Extraction

from config import Settings
from ingest.embedding import Embedder
from storage import ZvecStore

settings = Settings()
embedder = Embedder(settings.embed_model)
store = ZvecStore(settings.zvec_db_path)

EXTRACTION_PROMPT = """
Extract the following structured information from the text:

- decision: A conclusion or choice made during a meeting.
- action_item: A task assigned to someone.
- owner: The person responsible for an action item.
- deadline: A due date or time constraint mentioned.
- open_question: An unresolved question raised during the meeting.
- risk: A potential problem or concern flagged.

Do not hallucinate missing fields.
Only extract if clearly supported by the text.
"""

_EXAMPLE_TEXT = (
    "## Decisions\n"
    "- Keep the current signup copy for one more week.\n\n"
    "## Action Items\n"
    "- Priya: share final onboarding metrics by Monday.\n\n"
    "## Risks\n"
    "- Third-party API rate limits may affect daily sync jobs."
)

EXTRACTION_EXAMPLES = [
    ExampleData(
        text=_EXAMPLE_TEXT,
        extractions=[
            Extraction(
                extraction_class="decision",
                extraction_text="Keep the current signup copy for one more week.",
                char_interval=CharInterval(start_pos=18, end_pos=65),
            ),
            Extraction(
                extraction_class="action_item",
                extraction_text="Priya: share final onboarding metrics by Monday.",
                char_interval=CharInterval(start_pos=86, end_pos=134),
                attributes={"owner": "Priya", "deadline": "Monday"},
            ),
            Extraction(
                extraction_class="risk",
                extraction_text=(
                    "Third-party API rate limits may affect daily sync jobs."
                ),
                char_interval=CharInterval(start_pos=146, end_pos=201),
            ),
        ],
    )
]


def extract_actions(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search relevant chunks, then extract structured actions via LangExtract.
    """
    # 1) Search relevant chunks
    query_vector = embedder.embed([query])[0]
    search_results = store.search(query_vector, top_k=top_k)

    source_chunks: List[Dict[str, Any]] = []
    texts: List[str] = []
    for doc in search_results:
        fields = dict(doc.fields) if getattr(doc, "fields", None) else {}
        texts.append(fields.get("text", ""))
        source_chunks.append(
            {
                "id": doc.id,
                "score": doc.score,
                "metadata": fields,
            }
        )

    combined_context = "\n\n".join(texts)

    # 2) Call LangExtract with prompt_description and examples
    result = lx.extract(
        text_or_documents=combined_context,
        prompt_description=EXTRACTION_PROMPT,
        examples=EXTRACTION_EXAMPLES,
        model_id="gemini-3-flash-preview",
    )

    # 3) Normalize result to list of AnnotatedDocument
    if isinstance(result, list):
        annotated_docs = result
    else:
        annotated_docs = [result]

    # 4) Convert extractions into structured output
    structured: List[Dict[str, Any]] = []
    for annotated_doc in annotated_docs:
        for extraction in annotated_doc.extractions:
            entry: Dict[str, Any] = {
                "type": extraction.extraction_class,
                "text": extraction.extraction_text,
                "start_char": None,
                "end_char": None,
                "attributes": extraction.attributes,
            }
            if extraction.char_interval is not None:
                entry["start_char"] = extraction.char_interval.start_pos
                entry["end_char"] = extraction.char_interval.end_pos
            structured.append(entry)

    return {
        "query": query,
        "structured_output": structured,
        "source_chunks": source_chunks,
    }
