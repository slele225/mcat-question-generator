# MCAT Project

## Setup with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. Use it to create a virtual environment and install dependencies.

### Prerequisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `pip install uv` or the official installer).

### Run

From the project root:

```bash
uv venv
uv pip install -r requirements.txt
```

Activate the virtual environment:

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.\.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

Then run the project scripts as usual (e.g. `python generate_bank.py`).

---

## Question bank generation

- **Science:** One generation unit (one prompt) can produce **multiple** science questions. Use `--science-num-questions` to control how many questions are requested per science prompt. A single run over one unit may therefore append **several** science entries to `question_bank.jsonl`.
- **CARS:** Each generation unit still corresponds to one passage and one question set. One prompt writes **one** set to `question_bank.jsonl`.

---

## Directory structure

```
mcat project/
├── data/          # Data files (e.g. question banks, progress, JSON inputs)
├── scripts/       # Runnable scripts and entry points
├── src/           # Source packages and shared library code
├── requirements.txt
└── README.md
```

| Directory | Purpose |
|-----------|--------|
| `data/`   | Input/output data, JSON(L) files, question banks. |
| `scripts/`| CLI scripts and entry points meant to be run directly. |
| `src/`    | Reusable Python packages and library code. |

---

## License

See repository or author for license terms.
