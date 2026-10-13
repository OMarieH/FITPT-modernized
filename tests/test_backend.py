import json
import unittest
from types import SimpleNamespace

from modernized_rag import RagBackend, SourceDocument, WorkoutRequest, chunk_documents


class FakeEmbeddings:
    def __init__(self):
        self.inputs = []

    def create(self, *, model, input):
        self.inputs.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i, _ in enumerate(input)]
        )


class FakeChatCompletions:
    def __init__(self):
        self.messages = None

    def create(self, **kwargs):
        self.messages = kwargs["messages"]
        payload = {
            "title": "Beginner Gym Strength Session",
            "safety_note": "Stop if you feel pain.",
            "exercises": [
                {
                    "name": "Supported Row",
                    "sets": 3,
                    "reps": "8-10",
                    "rest_seconds": 60,
                    "instructions": "Keep the torso stable.",
                    "modification": "Use a lighter resistance.",
                }
            ],
            "journal_questions": ["How did the movement feel?"],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


class FakeOpenAI:
    def __init__(self):
        self.embeddings = FakeEmbeddings()
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class FakeIndex:
    def __init__(self):
        self.upserted = []
        self.query_calls = []

    def upsert(self, *, vectors, namespace):
        self.upserted.extend(vectors)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "matches": [
                {
                    "id": "chunk-1",
                    "metadata": {
                        "source": "https://example.org/strength",
                        "title": "Strength Basics",
                        "text": "Use controlled repetitions and suitable resistance.",
                        "position": 0,
                    },
                }
            ]
        }


class BackendTests(unittest.TestCase):
    def test_chunker_indexes_all_chunks_from_all_documents(self):
        documents = [
            SourceDocument("source-a", "one two three four five six", "A"),
            SourceDocument("source-b", "seven eight nine ten eleven twelve", "B"),
        ]
        chunks = chunk_documents(documents, chunk_words=4, overlap_words=1)

        self.assertEqual(4, len(chunks))
        self.assertEqual({"source-a", "source-b"}, {chunk.source for chunk in chunks})
        self.assertEqual(4, len({chunk.id for chunk in chunks}))

    def test_ingest_upserts_every_chunk_with_stable_ids(self):
        client, index = FakeOpenAI(), FakeIndex()
        backend = RagBackend(client, index)
        documents = [SourceDocument("source-a", " ".join(f"word{i}" for i in range(400)))]

        first_count = backend.ingest(documents)
        first_ids = [vector["id"] for vector in index.upserted]
        index.upserted.clear()
        second_count = backend.ingest(documents)
        second_ids = [vector["id"] for vector in index.upserted]

        self.assertGreater(first_count, 1)
        self.assertEqual(first_count, second_count)
        self.assertEqual(first_ids, second_ids)

    def test_retrieval_uses_complete_user_request_not_generic_question(self):
        client, index = FakeOpenAI(), FakeIndex()
        backend = RagBackend(client, index)
        request = WorkoutRequest("gym", "upper-body strength", "beginner", 30, ("wrist pain",))

        backend.retrieve(request)

        embedded_query = client.embeddings.inputs[-1][0]
        self.assertIn("upper-body strength", embedded_query)
        self.assertIn("wrist pain", embedded_query)
        self.assertNotEqual("question", embedded_query)

    def test_generation_validates_structure_and_attaches_retrieved_sources(self):
        backend = RagBackend(FakeOpenAI(), FakeIndex())
        request = WorkoutRequest("gym", "strength", "beginner", 30)

        plan = backend.generate(request)

        self.assertEqual("Supported Row", plan.exercises[0].name)
        self.assertEqual("https://example.org/strength", plan.sources[0].url)


if __name__ == "__main__":
    unittest.main()
