"""Condensed poster excerpt; production implementation is in modernized_rag/."""

request = WorkoutRequest(
    location="gym",
    goal="upper-body strength",
    experience_level="beginner",
    duration_minutes=30,
    limitations=("wrist discomfort",),
)

# INGESTION: index every chunk, not only documents[0].
chunks = chunk_documents(source_documents, chunk_words=180, overlap_words=30)
embeddings = openai.embeddings.create(
    model="text-embedding-3-small",
    input=[chunk.text for chunk in chunks],
)
index.upsert(
    namespace="fitness-knowledge-v2",
    vectors=[
        {
            "id": chunk.id,  # Deterministic IDs prevent duplicate reruns.
            "values": vector.embedding,
            "metadata": {
                "text": chunk.text,
                "source": chunk.source,
                "title": chunk.title,
            },
        }
        for chunk, vector in zip(chunks, embeddings.data)
    ],
)

# RETRIEVAL: search with the complete request instead of "question".
query_vector = openai.embeddings.create(
    model="text-embedding-3-small",
    input=[request.retrieval_query()],
).data[0].embedding
evidence = index.query(
    namespace="fitness-knowledge-v2",
    vector=query_vector,
    top_k=4,
    include_metadata=True,
)

# GENERATION: retrieved text constrains a validated, structured response.
plan = generate_structured_workout(request=request, evidence=evidence.matches)
print(plan.to_dict())
