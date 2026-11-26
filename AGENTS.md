# Working on rag-demo-system

Start in README.md. The four command examples are the public teaching flow. Run `./demo --help` for the CLI surface. Use the project .venv for Python packages.

- `demo` bootstraps the project's environment and launches the client.
- `app.py` links the model wizard, RAGFlow API adapter, file browser, ingestion, retrieval, chat and deletion menus. The staged CLI and interactive menu share their operations. Dataset IDs scope every document and search.
- `iterative.py` decides whether to search again and combines ranked passages across passes. Keep raw search free of generation calls and bound the context presented to the model.
- `epub.py` handles the specific native OPF encoding diagnostic. Keep source documents byte-for-byte intact and repair into a new file.
- `sessions.py` stores opt-in transcripts in SQLite, scoped to backend URL and dataset ID. A saved transcript includes copies of cited excerpts.
- `monitoring/` contains an optional standalone Grafana, Prometheus and TCP probe deployment.
- `tests/` runs without a service, key or GPU. Run `./demo --help`, then `.venv/bin/python -m unittest discover -s tests -v`.

Embedding registration belongs to the RAGFlow instance. Bind the embedding model when creating a collection. Read the chat model from client settings when making completion requests. The README links the upstream API revision used for live verification.

For end-to-end verification, create a uniquely named disposable collection, ingest examples/station.txt, inspect the chunks, ask when Aurora opened and check both the year and cited passage. Exercise deletion on that collection. Keep credentials in environment variables and local endpoint settings outside the checkout. Publish only synthetic fixtures.
