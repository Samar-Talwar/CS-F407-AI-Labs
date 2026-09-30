# Week 04 – Logic / Planning

## How to run

```bash
# 1. Activate the venv (if not already active)
#    .venv\Scripts\Activate.ps1  (PowerShell)
#    source .venv/bin/activate   (bash/zsh)

# 2. Install dependencies (if needed)
#    pip install -e .

# 3. Regenerate all results and print summaries
python -m week04_logic.src.cli --regenerate

# 4. Just print summaries (if results already exist)
python -m week04_logic.src.cli --summary

# 5. Run the test suite
pytest week04_logic -q
```

## Prolog dependency

The optional Prolog extension requires [SWI-Prolog](https://www.swi-prolog.org/) to be installed and available on your PATH as `swipl`.

If `swipl` is not found, the Prolog tests will be skipped with a visible reason (not silently pass).