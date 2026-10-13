"""Command-line entry point for the modernized FITPT backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from modernized_rag import RagBackend, WorkoutRequest
from modernized_rag.backend import EMBEDDING_DIMENSION
from modernized_rag.sources import fetch_documents, read_urls


INDEX_NAME = "fitpt-rag-v2"


def build_backend() -> RagBackend:
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY before running FITPT")

    openai_client = OpenAI()
    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pinecone.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    return RagBackend(openai_client, pinecone.Index(INDEX_NAME))


def main() -> None:
    parser = argparse.ArgumentParser(description="FITPT modernized RAG backend")
    parser.add_argument("command", choices=("ingest", "generate"))
    parser.add_argument("--corpus", default="Corpus.txt")
    parser.add_argument("--location", default="gym")
    parser.add_argument("--goal", default="upper-body strength")
    parser.add_argument("--level", default="beginner")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--limitation", action="append", default=[])
    args = parser.parse_args()

    backend = build_backend()
    if args.command == "ingest":
        documents = fetch_documents(read_urls(Path(args.corpus)))
        print(json.dumps({"documents": len(documents), "chunks_indexed": backend.ingest(documents)}))
        return

    request = WorkoutRequest(
        location=args.location,
        goal=args.goal,
        experience_level=args.level,
        duration_minutes=args.minutes,
        limitations=tuple(args.limitation),
    )
    print(json.dumps(backend.generate(request).to_dict(), indent=2))


if __name__ == "__main__":
    main()
