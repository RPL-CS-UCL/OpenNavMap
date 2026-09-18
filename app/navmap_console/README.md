# navmap_console

Web console for OpenNavMap multi-session mapping: register or upload session submaps,
run and append incremental map merges, inspect the intermediate results interactively,
evaluate, and export a LiteVLoc navigation map. It replaces the
`scripts/run_map_merging.sh` workflow for day-to-day use.

## Layout

```
backend/navmap_console/   FastAPI package (Python 3.8, conda env `opennavmap`)
backend/tests/            pytest suite (fixtures use python/visualization/example_data/synthetic_map)
frontend/                 Vite + React + TypeScript + Tailwind + shadcn/ui single-page app
scripts/dev.sh            backend with reload (8765) + Vite dev server (5173, proxies /api and /ws)
scripts/build.sh          pnpm install + pnpm build -> frontend/dist
scripts/serve.sh          production start: conda python -m uvicorn, serves frontend/dist
```

## Install

The backend dependencies are already in the `opennavmap` conda environment
(`backend/requirements-console.txt` lists them with their Python 3.8 ceilings).
The frontend needs Node >= 20 and pnpm:

```bash
cd frontend && pnpm install
```

## Run

```bash
# development: http://localhost:5173 (hot reload), API on :8765
bash scripts/dev.sh

# production: build once, then serve everything from :8765
bash scripts/build.sh
bash scripts/serve.sh
```

API docs are served at `/api/docs`.

## Configuration

Every variable has a default; set them in the shell before `serve.sh` / `dev.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `NAVMAP_CONSOLE_DATA_ROOT` | `/Titan/dataset/data_opennavmap/navmap_console` | Where regions, runs, jobs, uploads live |
| `NAVMAP_CONSOLE_HOST` | `0.0.0.0` | Bind address |
| `NAVMAP_CONSOLE_PORT` | `8765` | Backend port |
| `NAVMAP_CONSOLE_REPO` | inferred from the package location | OpenNavMap repository root |
| `NAVMAP_CONSOLE_PYTHON` | `/root/miniconda3/envs/opennavmap/bin/python` | Interpreter for merge / preprocess subprocesses |
| `NAVMAP_CONSOLE_EVAL_PYTHON` | `/root/miniconda3/envs/traj_evaluation/bin/python` | Interpreter for the official trajectory evaluation |
| `NAVMAP_CONSOLE_ALLOWED_ROOTS` | `/Titan/dataset` | Colon-separated roots the server-path browser may list |
| `MERGE_CPU_LIST` | auto-detected non-boost cores | `taskset -c` list for merge subprocesses; empty string disables pinning |
| `CUDA_VISIBLE_DEVICES` | unset | Passed through to subprocesses |

## Data directory

```
<data_root>/
├── regions/<region_id>/
│   ├── region.json               display name, VPR config, current head (run + step)
│   ├── map -> runs/<run_id>/final symlink to the current final map
│   ├── sessions/<session_id>/    session.json (+ data/ for uploaded sessions)
│   ├── runs/<run_id>/            run.json, steps.json, inputs/, base/, output/, final/, cache/, evaluations/
│   └── exports/<export_id>/      export.json + navigation-map zip
├── jobs/<job_id>.json            subprocess record; <job_id>.log is its merged stdout/stderr
└── uploads/                      temporary upload files
```

Every object is a JSON file; nothing lives in a database, so `ls` shows the state and a
backend restart loses nothing. `run.json`, `steps.json` and `jobs/*.json` are written
atomically (temp file + rename).

## Network access

There is no login. The console is meant for the lab LAN only: it can browse server
directories under `NAVMAP_CONSOLE_ALLOWED_ROOTS`, delete sessions and runs, and start
GPU jobs. Put it behind a reverse proxy with authentication before exposing it further.
Merges started outside the console (for example with `scripts/run_map_merging.sh`) are not
scheduled by it and will contend for the GPU.

## Tests

```bash
# backend
cd backend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/miniconda3/envs/opennavmap/bin/python -m pytest tests -q

# frontend (vitest + jsdom), type check + build, and the .gitignore guard
cd frontend
pnpm test
pnpm build
pnpm check:ignored
```

## File naming guard

The repository `.gitignore` has broad patterns (`*vis*`, `log*`, `output*`, `paper*`,
`cache/`, `tmp/`, `dataset/`). `app/navmap_console/` is re-included as a whole, but keep
these out of new file and directory names anyway so the guard never has to fight the
rules: any name containing `vis` (including `visible`, `revision`, `advisor`), names
starting with `log`, `output` or `paper`, and directories named `dataset`, `cache` or
`tmp`. `pnpm check:ignored` fails the build if a frontend source file is ignored.
