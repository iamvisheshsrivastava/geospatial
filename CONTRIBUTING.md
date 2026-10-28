# Contributing

Thanks for taking a look at this project. It's a PyTorch + FastAPI platform for
satellite imagery and LiDAR analysis — see `README.md` for what the API does.
This guide covers getting set up, running it, testing it, and submitting
changes.

## Dev setup

Requires Python 3.11.

```bash
git clone https://github.com/iamvisheshsrivastava/geospatial.git
cd geospatial
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You do **not** need trained model checkpoints to work on most of the codebase.
The API starts up fine without them — endpoints that depend on a missing
checkpoint (`/predict`, `/anomaly`, `/segment`, ...) return a clean `503`
instead of crashing, and most tests mock the model layer out entirely (see
`tests/test_api.py`). You only need real checkpoints if you're working on
training itself (`src/train.py`, `src/anomaly.py`, the notebooks).

Do **not** touch anything related to the Kaggle API token / checkpoint
download pipeline (`.github/workflows/ci.yml`'s Kaggle step, `KAGGLE_API_TOKEN`)
unless a maintainer asks — that's deployment infrastructure, not part of a
normal contribution.

## Running it locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- `http://localhost:8000` — browser UI
- `http://localhost:8000/docs` — Swagger UI (interactive API docs)

Or via Docker, which matches the production image more closely:

```bash
docker compose up --build
```

Quick smoke test once it's running:

```bash
curl http://localhost:8000/health
```

## Running tests

The test suite uses `pytest` and mocks out the heavy ML dependencies (model
weights, GPU inference) wherever possible, so it runs fast and needs no
checkpoints or GPU:

```bash
PYTHONPATH=. pytest -q
```

Run a single file or test while iterating:

```bash
PYTHONPATH=. pytest tests/test_api.py -k test_predict_503_when_classifier_missing -q
```

This is exactly what CI runs (`.github/workflows/ci.yml`, `test` job) before
anything is deployed — a PR with failing tests won't be merged.

### Adding tests

If you touch `src/api/main.py`, add or update a test in `tests/test_api.py`.
Look at existing tests (e.g. `test_predict_503_when_classifier_missing`,
`test_anomaly_503_when_autoencoder_missing`) for the pattern: use FastAPI's
`TestClient`, monkeypatch the relevant global/model or the `src.*` function
the endpoint calls, and assert on status code + response shape. You don't
need a real model checkpoint for this.

## Code style

There's no enforced linter/formatter in CI yet, so match the style already in
the file you're editing: type hints on function signatures, module/function
docstrings explaining the *why* (not just the *what* — see the comments in
`src/api/main.py` for the convention), and small, single-purpose functions.

## Pull request process

1. Fork the repo and create a branch off `main` (e.g. `fix/change-detect-test`).
2. Make your change. Keep PRs scoped to one issue/fix — easier to review, and
   `git bisect`-friendly if something regresses later.
3. Add or update tests covering your change. A PR that fixes a bug without a
   regression test for it will likely get a request to add one.
4. Run `PYTHONPATH=. pytest -q` locally and make sure it's green.
5. Open the PR against `main`. Reference the issue it closes (e.g.
   `Closes #39`). Describe what changed and why, not just what.
6. CI runs the test suite automatically on the PR. The `deploy` job only runs
   on pushes to `main` and needs repo secrets, so it won't run on your fork's
   PR branch — that's expected, it only matters for the test job passing.
7. Address review feedback with new commits (no need to force-push/squash
   until a maintainer asks).

## Reporting bugs / proposing changes

Open a GitHub issue before starting on anything non-trivial, so the approach
can be discussed first. Issues labeled `good first issue` are scoped to be
small, self-contained, and don't require trained model checkpoints or
Kaggle/data setup — a good place to start if you're new to the repo.
