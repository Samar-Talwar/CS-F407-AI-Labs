# Week 01 – Neural Models

**How to run**

```bash
# Run pytest tests (works directly from repo root without pip install -e)
pytest week01_neural_models/

# Regenerate all results into results/
python -m week01_neural_models.src.cli
```

The CLI writes JSON files into `results/`. All numbers reported in `REPORT.md` come from these files.

**Project layout**

```
week01_neural_models/
├── src/                 # importable modules + CLI entry point
│   ├── __init__.py
│   ├── models.py        # neural network definitions (XORBinaryNet, XORLinearNet, XORMulticlassNet)
│   ├── train.py         # training loops + diagnostics (binary, linear baseline, symmetry, multiclass, seed sweep)
│   └── cli.py           # reproduces every result from the lab sheet
├── tests/               # pytest suite – every claim and task has a test
│   └── test_week01.py
├── results/             # machine‑generated outputs
│   ├── linear_baseline.json
│   ├── binary_xor.json
│   ├── symmetry.json
│   ├── activation_comparison.json
│   ├── multiclass.json
│   └── seed_sweep.json
├── CHECKLIST.md         # mapping of every requirement to proof
├── REPORT.md            # filled‑in submission (numbers from results/)
└── README.md            # this file
```
