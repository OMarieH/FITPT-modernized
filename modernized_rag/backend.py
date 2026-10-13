"""Core ingestion, retrieval, and workout-generation logic for FITPT.

The API clients are injected so the important backend behavior can be tested
without network access or paid API calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Sequence


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"
DEFAULT_NAMESPACE = "fitness-knowledge-v2"


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class TextChunk:
    id: str
    source: str
    text: str
    title: str
    position: int


@dataclass(frozen=True)
class WorkoutRequest:
    location: str
    goal: str
    experience_level: str
    duration_minutes: int
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("location", "goal", "experience_level"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if not 5 <= self.duration_minutes <= 180:
            raise ValueError("duration_minutes must be between 5 and 180")

    def retrieval_query(self) -> str:
        limitations = ", ".join(self.limitations) if self.limitations else "none reported"
        return (
            f"Evidence-based {self.experience_level} workout for {self.location}. "
            f"Goal: {self.goal}. Duration: {self.duration_minutes} minutes. "
            f"Limitations: {limitations}. Include safe exercise selection, form, "
            "sets, repetitions, rest, and appropriate modifications."
        )


@dataclass(frozen=True)
class Exercise:
    name: str
    sets: int
    reps: str
    rest_seconds: int
    instructions: str
    modification: str


@dataclass(frozen=True)
class SourceCitation:
    title: str
    url: str


@dataclass(frozen=True)
class WorkoutPlan:
    title: str
    safety_note: str
    exercises: tuple[Exercise, ...]
    journal_questions: tuple[str, ...]
    sources: tuple[SourceCitation, ...]

    @classmethod
    def from_model_json(
        cls, raw_json: str, retrieved_chunks: Sequence[TextChunk]
    ) -> "WorkoutPlan":
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("The model did not return valid JSON") from exc

        required = ("title", "safety_note", "exercises", "journal_questions")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Model output is missing fields: {', '.join(missing)}")
        if not isinstance(payload["exercises"], list) or not payload["exercises"]:
            raise ValueError("Model output must contain at least one exercise")
        if not isinstance(payload["journal_questions"], list):
            raise ValueError("journal_questions must be a list")

        exercises: list[Exercise] = []
        exercise_fields = {
            "name", "sets", "reps", "rest_seconds", "instructions", "modification"
        }
        for item in payload["exercises"]:
            if not isinstance(item, dict) or not exercise_fields.issubset(item):
                raise ValueError("Each exercise must contain all required fields")
            exercises.append(
                Exercise(
                    name=str(item["name"]),
                    sets=int(item["sets"]),
                    reps=str(item["reps"]),
                    rest_seconds=int(item["rest_seconds"]),
                    instructions=str(item["instructions"]),
                    modification=str(item["modification"]),
                )
            )

        citations: list[SourceCitation] = []
        seen_sources: set[str] = set()
        for chunk in retrieved_chunks:
            if chunk.source not in seen_sources:
                citations.append(SourceCitation(title=chunk.title or chunk.source, url=chunk.source))
                seen_sources.add(chunk.source)

        return cls(
            title=str(payload["title"]),
            safety_note=str(payload["safety_note"]),
            exercises=tuple(exercises),
            journal_questions=tuple(str(q) for q in payload["journal_questions"]),
            sources=tuple(citations),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_documents(
    documents: Iterable[SourceDocument], chunk_words: int = 180, overlap_words: int = 30
) -> list[TextChunk]:
    """Split every source document and assign deterministic, rerun-safe IDs."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    step = chunk_words - overlap_words
    chunks: list[TextChunk] = []
    for document in documents:
        words = document.text.split()
        for position, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + chunk_words]).strip()
            if not text:
                continue
            identifier = sha256(
                f"{document.source}\n{position}\n{text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                TextChunk(
                    id=identifier,
                    source=document.source,
                    text=text,
                    title=document.title,
                    position=position,
                )
            )
            if start + chunk_words >= len(words):
                break
    return chunks


class RagBackend:
    def __init__(
        self,
        openai_client: Any,
        vector_index: Any,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        embedding_model: str = EMBEDDING_MODEL,
        chat_model: str = CHAT_MODEL,
    ) -> None:
        self.openai = openai_client
        self.index = vector_index
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.openai.embeddings.create(model=self.embedding_model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def ingest(self, documents: Iterable[SourceDocument]) -> int:
        """Embed and upsert every chunk. Stable IDs prevent duplicate reruns."""
        chunks = chunk_documents(documents)
        embeddings = self._embed([chunk.text for chunk in chunks])
        vectors = [
            {
                "id": chunk.id,
                "values": embedding,
                "metadata": {
                    "source": chunk.source,
                    "title": chunk.title,
                    "text": chunk.text,
                    "position": chunk.position,
                },
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        for start in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[start : start + 100], namespace=self.namespace)
        return len(vectors)

    def retrieve(self, request: WorkoutRequest, top_k: int = 4) -> list[TextChunk]:
        query_vector = self._embed([request.retrieval_query()])[0]
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        matches = result.matches if hasattr(result, "matches") else result["matches"]
        chunks: list[TextChunk] = []
        for match in matches:
            metadata = match.metadata if hasattr(match, "metadata") else match["metadata"]
            match_id = match.id if hasattr(match, "id") else match["id"]
            chunks.append(
                TextChunk(
                    id=match_id,
                    source=str(metadata["source"]),
                    title=str(metadata.get("title", "")),
                    text=str(metadata["text"]),
                    position=int(metadata.get("position", 0)),
                )
            )
        return chunks

    def generate(self, request: WorkoutRequest, top_k: int = 4) -> WorkoutPlan:
        chunks = self.retrieve(request, top_k=top_k)
        if not chunks:
            raise RuntimeError("No fitness knowledge was retrieved")
        context = "\n\n".join(
            f"SOURCE {number}: {chunk.title or chunk.source}\n"
            f"URL: {chunk.source}\n{chunk.text}"
            for number, chunk in enumerate(chunks, start=1)
        )
        response = self.openai.chat.completions.create(
            model=self.chat_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fitness-planning assistant. Use only the supplied "
                        "evidence for fitness claims. Do not diagnose or treat medical "
                        "conditions. Return JSON with title, safety_note, exercises, and "
                        "journal_questions. Each exercise requires name, sets, reps, "
                        "rest_seconds, instructions, and modification."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"REQUEST\n{request.retrieval_query()}\n\n"
                        f"RETRIEVED EVIDENCE\n{context}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The model returned an empty response")
        return WorkoutPlan.from_model_json(content, chunks)
