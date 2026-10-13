# FITPT RAG Backend — Poster Material

## Backend pipeline

```text
Curated fitness URLs
        ↓
Fetch and clean documents
        ↓
Chunk every document + assign stable IDs
        ↓
OpenAI embeddings → Pinecone vector index
        ↓
Embed the complete user workout request
        ↓
Retrieve relevant fitness evidence
        ↓
Generate + validate structured workout JSON
        ↓
Workout, journal questions, and source citations
```

## Defect found in the original prototype

The original ingestion expression was:

```python
[documents[0].page_content for t in docs]
```

It iterated through the unsplit sources while repeatedly selecting only the first
chunk. With one source document, only one chunk entered Pinecone. With multiple
sources, the same first chunk would be duplicated.

The program still returned a plausible workout because the vector store was not
empty and GPT-3.5 could create a workout from pretrained knowledge. Plausible text
therefore masked a retrieval failure.

The corrected implementation:

- chunks every source document;
- embeds and upserts every chunk;
- uses deterministic IDs, making repeat ingestion idempotent;
- searches with the user's complete goal, location, duration, experience, and limitations;
- validates the output schema; and
- attaches citations from retrieved metadata rather than asking the model to invent them.

## Concise poster description

**FITPT is a retrieval-augmented workout-planning backend.** It converts curated
fitness resources into embeddings stored in a vector database. A user's complete
workout request is embedded to retrieve relevant guidance, which is supplied to a
language model as constrained context. The backend validates the generated workout
and returns exercises, modifications, journal prompts, and traceable sources.

Use `poster/code_excerpt.py` as the compact code panel. It intentionally shows the
three most important backend stages; `modernized_rag/backend.py` contains the full
implementation and validation logic.

## Suggested presenter talking point

> My original prototype produced convincing workouts, but convincing output did not
> prove that retrieval worked. I found that the ingestion code indexed only the first
> chunk. I redesigned the backend so every chunk is indexed with a stable identifier,
> retrieval uses the actual user profile, and the final response is structured,
> validated, and linked to its retrieved sources.
