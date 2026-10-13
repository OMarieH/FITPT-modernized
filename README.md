# FITPT

FITPT is a retrieval-augmented workout-planning capstone project.

The original May 2024 Colab notebook is preserved unchanged in
`Capstone_FITPT.ipynb`. The modernized backend is implemented separately in
`modernized_rag/` so the original prototype and the corrected design can be
compared directly.

## Modernized backend

The corrected pipeline fetches the original URL corpus, chunks every document,
stores embeddings in Pinecone with deterministic IDs, retrieves evidence using
the complete workout request, and validates structured workout output.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# Add credentials to your shell or secret manager; never commit them.
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."

python run_rag.py ingest
python run_rag.py generate \
  --location gym \
  --goal "upper-body strength" \
  --level beginner \
  --minutes 30
```

Run the offline backend tests without API keys:

```bash
python -m unittest discover -s tests -v
```

Poster-ready explanations and an example response are in `poster/`.
