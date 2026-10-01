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
## Authorship
- Every source file starts with: # CS F407 Lab, Week N | Author: Samar Talwar | Not licensed for reuse or submission by others.
- Never add an open-source license for original code. NOTICE.md governs it.

## Quality gates (mandatory, learned from the Week 1 audit)
1. Fresh-clone gate: after committing, run `git clone . $env:TEMP\fresh_check` (delete any old copy first),
   then in that clone run ruff check ., pytest -q and the week's CLI using the existing venv python,
   WITHOUT pip install -e. All must pass before pushing.
2. Record-everything: every value the sheet says to record, report, predict, print, verify or compare
   (losses, shapes, counts, paths, expanded nodes, probabilities, etc.) is written to a JSON/CSV in results/
   at full precision, with a clear key. CHECKLIST.md rows must name the exact file and key.
3. Negative and baseline cases the sheet asks for (impossible inputs, trivial cases, baseline models) each get
   their own experiment, their own result file and their own test.
4. Real mutation tests: for each core algorithm add a test that breaks the REAL module
   (monkeypatch the actual function) and asserts that the outcome fails. Do not re-implement the model inside the test.
5. No self-comparison: an equality check must compare two independently computed quantities.
   Use an independent oracle where possible (e.g. Dijkstra vs BFS vs A* on the same input).
6. Honest reporting: report stochastic outcomes as actual counts over seeds. Never claim more than measured.
   Do not write "100% coverage" or similar unless measured by a tool.
7. Do not round in results files. Rounding is for the report only.
8. Finish by pasting real command output (pytest summary line, ruff result), the pushed commit hash,
   and confirming `git status` is clean and HEAD equals origin/main.
9. One audit pass only. Do not rewrite code that already passes.
## Notebook weeks (5, 6, 7)
- Never print a notebook whole. Extract it with a short Python script (json module) into a scratch file outside the repo
  (%TEMP%\nb_dump_weekNN.txt): for each cell print the index, type and source; for outputs keep only text/plain truncated to
  400 characters; never print image or base64 data. Read the dump in chunks of about 150 lines.
- CHECKLIST.md lists every task, exercise, question, TODO cell and printed result in the sources, each mapped to the file,
  results key or test that satisfies it. Do not spend time on cleaned-up notebooks: the deliverable is src, tests, results, report.
- Dependencies: use only packages already in the venv (numpy, torch, matplotlib) plus the standard library. Do not download
  models, call paid APIs or add heavy dependencies. Anything else is import-guarded and optional.
- Every LLM or embedding call goes behind a small interface with a deterministic offline stub (used by all tests and the default CLI)
  and an optional real backend (for example local Ollama) enabled only by a flag and detected at runtime.
- NEVER fabricate LLM outputs. Stub outputs are labelled "stub" in every results file and in the report. Saved outputs from the
  professor's notebook go to results/original_notebook_outputs.json, labelled as such and kept separate from our reproductions.
- Tests needing a real backend are skipped with a visible reason when it is absent.
- Never edit anything in original/. README states provenance and the Apache-2.0 course repo attribution.
- Testing standard: independent oracle written in the test file (brute-force or NumPy reference), normalisation checks, at least
  five hand-checkable values, error-handling cases, and three real mutation tests that monkeypatch the actual src functions.
- Fresh-clone gate on Windows: git clone . $env:TEMP\fresh_wNN, then ruff check ., pytest -q -rs and the week's CLI using
  C:\Users\Samar\documents\CS-F407-AI-Labs\.venv\Scripts\python.exe, without pip install -e.