# Week 01 – Neural Models

**How to run**

```bash
# Install dependencies (once)
pip install -e .

# Regenerate all results
python -m week01_neural_models.src.cli
```

The CLI writes JSON files into `results/`. All numbers reported in `REPORT.md` come from these files.

**Project layout**

```
week01_neural_models/
├── src/                 # importable modules + CLI entry point
│   ├── __init__.py
│   ├── models.py        # neural network definitions
│   ├── train.py         # training loops + diagnostics
│   └── cli.py           # reproduces every result from the lab sheet
├── tests/               # pytest suite – every claim has a test
│   └── test_week01.py
├── results/             # machine‑generated outputs (git‑ignored)
│   ├── binary_xor.json
│   ├── symmetry.json
│   ├── activation_comparison.json
│   └── multiclass.json
├── CHECKLIST.md         # mapping of every requirement to proof
├── REPORT.md            # filled‑in submission (numbers from results/)
└── README.md            # this file
```