# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repository is

Solutions to the assignments of **EV – Evoluční výpočetní techniky** (UTB ve Zlíně, FAI).
Each assignment is handed in to the teacher as a folder that contains both runnable code
and the written report, so the folder must make sense on its own — also when it is
downloaded as a ZIP, without the rest of the repository.

## Layout

```
NN-nazev/                # one folder per assignment: 01-genetic-algorithm, 02-…, 03-…
├── README.md            # the report (Czech) – task, implementation, results, conclusions
├── main.py              # CLI entry point for the experiments
├── ga/ | de/            # the algorithm itself, independent of the test problems
├── problems/            # objective functions
├── experiments/         # repeated runs, statistics, plots, tables, parameter studies
├── tests/               # pytest
└── results/             # generated plots and tables – committed, they are part of the submission
```

Shared for the whole repository: `requirements.txt`, `.venv/` (ignored), `run_tests.sh`,
and the assignment table in `README.md`.

## Conventions for a new assignment

- Copy the structure of `02-real-parameter-ga` or `03-differential-evolution`; they are the
  reference layout.
- **Each assignment is self-contained.** Do not import across assignment folders and do not
  factor shared code out into a common package — the teacher checks one folder at a time.
  Duplicating `problems/` into a new assignment is correct here, not a mistake.
- Keep the algorithm independent of the problems: the solver works against a small interface
  (`dim`, `evaluate(points)`), so variants and problems can be combined freely.
- Experiments are reproducible: seeds are fixed (`np.random.default_rng(seed + run)`), so
  re-running `main.py` must regenerate byte-identical files in `results/`.
- Statistics over 10 independent runs (best, worst, mean, median, std) plus a convergence
  plot, unless the assignment says otherwise.
- Tuning and final runs use different seeds, so the reported numbers are not measured on the
  same runs the configuration was selected on.
- Every number quoted in a README must come from a file in `results/` — verify against the
  CSV, do not retype from memory.

## Language

- **Czech**: everything a human reads as part of the submission — `README.md` of the
  assignment, plot titles, axis labels and legends, column headers of result tables.
- **English**: code, identifiers, docstrings and code comments, commit messages, and this file.
  The whole existing codebase uses English comments; keep it consistent.

## Before opening a pull request

1. `./run_tests.sh` — runs the tests of every assignment in its own process (the assignments
   deliberately reuse module names such as `ga`, `problems`, `experiments`, so a single
   `pytest` run from the repository root cannot collect them together). All tests must pass.
2. Regenerate the results of the assignment you changed and check `git status` — anything
   unexpected means the run is not reproducible.
3. Update the assignment table in the root `README.md` when adding a new assignment.
4. Plain commit messages, no `Co-Authored-By` or other attribution trailers.

## Do not

- Do not touch the `results/` of an assignment you were not asked to work on; regenerating
  them produces diff noise in a submission that is already handed in.
- Do not add dependencies beyond `numpy`, `matplotlib` and `pytest` without asking.
- Do not commit `.venv/`, `__pycache__/` or `.DS_Store` (already covered by `.gitignore`).
