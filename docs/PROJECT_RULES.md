# CS F407 AI Labs: Project Rules

## Goal
One professional, reproducible repo holding solutions to every hands-on in CS F407
(BITS Pilani Goa, AY 2026-27 Sem 1). Structure mirrors the official course weeks.
Follow each lab sheet STRICTLY: every Task, Test, measurement, Think-About-It,
Reflection Question and Submission item must be addressed.

## Sources of truth
- Weeks 1,2,3,4,8: PDFs in docs/lab_sheets/
  (W1 neur_models_lab_ex, W2 agents_lab, W3 search_lab_ex, W4 logic_lab_ex, W8 BN_lab).
- Weeks 5,6,7: the professor's notebooks (.ipynb), saved UNMODIFIED in weekNN_*/original/.
  NEVER edit anything in original/.
- Before coding a week, read its sheet/notebook fully and write weekNN_*/CHECKLIST.md
  mapping every requirement to the file/test/result that satisfies it.

## Layout per week
weekNN_name/  src/  tests/  results/  README.md  REPORT.md  CHECKLIST.md
- src/: importable modules + a CLI entry point that reproduces every result.
- tests/: pytest. Every claim in the lab has a test
  (probabilities sum to 1, A* path is optimal, unsolvable input returns "no plan", etc.).
- results/: machine-generated outputs written by the CLI. REPORT.md numbers come from here.

## Code standards
- Python >= 3.10, type hints, docstrings on public functions, no wildcard imports,
  no global mutable state.
- Deterministic: every random source takes an explicit seed. CPU only.
- Efficient: right data structure for the job (heapq for A*, deque for BFS,
  frozenset states for planning, closed sets), vectorised numpy/torch, no recomputation.
- Respect each sheet's hard constraints
  (W1: PyTorch 2-2-1 net, BCEWithLogitsLoss, full batch, random init;
   W3: implement A* ourselves, no search libraries;
   W8: plain Python data structures, no ML library, no pretrained LM).
- `ruff check .` and `pytest` must pass before any commit.

## Environment (Windows / PowerShell)
- All files UTF-8 WITHOUT BOM, LF line endings. Never use `Set-Content -Encoding utf8` in
  Windows PowerShell 5.1 (it adds a BOM and breaks TOML). Prefer Python or the editor tools.
- Activate venv: .venv\Scripts\Activate.ps1

## Honesty rules
- NEVER fabricate results. Run the code; copy numbers from results/.
- Reflection, "in your own words" and "Think About It" answers: write a TODO(student) stub
  with 2-3 factual hints drawn from the actual results. The student writes the final text.
- Log every AI-assistant prompt that shapes the code in docs/prompt_log.md
  (date, week, prompt, what was changed after review).

## Workflow
- One week at a time. Propose a plan, wait for approval, then implement.
  Do not touch other weeks.
- Commit per logical unit with Conventional Commits (feat:, test:, docs:, chore:).
  Never commit .venv or large binaries.
- After a week: tick CHECKLIST.md, update the root README status table, list any gaps.