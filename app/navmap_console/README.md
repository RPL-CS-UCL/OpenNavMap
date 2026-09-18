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

## Runs and jobs

A **run** is one merge over an ordered list of sessions inside a region; the **job** behind
it is the pipeline subprocess. Runs live in `regions/<rid>/runs/<run_id>/` (`run.json`,
`steps.json`, `output/merge_*` step directories, `final/` after consolidation) and jobs in
`jobs/<job_id>.json` with the merged stdout/stderr in `jobs/<job_id>.log`. The subprocess
writes that log file directly; the backend only tails it, so nothing is lost if the backend
goes away.

Two queues, each running one job at a time:

| Queue | Jobs | Why one at a time |
|-------|------|-------------------|
| `gpu` | merge, append | the pipeline keeps the VPR and MASt3R models resident on the GPU |
| `cpu` | consolidate, official evaluation, result import, export packing and verification | disk and CPU bound; keeps them off the GPU queue |

When a merge succeeds the last step is consolidated into `final/` (image hard links, no extra
space) and promoted to the region head; `regions/<rid>/map` points at it.

**Restarts.** Stopping the backend does not stop a running merge. On startup every job
recorded as `running` is looked up by pid and process creation time: if the process is still
there the backend re-attaches and keeps tailing the log; if it is gone the job is judged by
its log tail (`STEP_DONE` / `merge_finalmap ->` in the last lines means it finished) and
otherwise marked `orphaned`. Log lines written while no backend was watching are replayed
from the byte offset stored in the job record, so step completions are never skipped.
Cancelling sends `SIGTERM` to the whole process group and escalates to `SIGKILL` after 10 s.

**Crash kinds.** A failed job gets a `crash_kind` and the UI shows a matching hint:

| Kind | Detected by | Meaning |
|------|-------------|---------|
| `segfault` | return code -11, "Segmentation fault" | native crash; on this host usually the degraded favoured cores, so re-run with a different `MERGE_CPU_LIST` |
| `bit_flip` | impossible Python errors such as `'range_iterator' object is not callable` | CPython frame corruption from a memory bit flip (see the repo `CLAUDE.md`); same remedy as `segfault` |
| `cuda_oom` | "CUDA out of memory" | free the GPU or lower the image size |
| `oom` | return code -9/137, "Killed", `MemoryError` | host RAM exhausted (the OOM killer uses `SIGKILL`) |
| `terminated` | return code -15/-2/143/130 without a cancel request | `SIGTERM`/`SIGINT` from outside the console |
| `error` | anything else | read the last lines of the log |

**Environment of the subprocess.** Copied from `scripts/run_map_merging.sh`: `LD_PRELOAD` is
pinned to the conda `libstdc++.so.6`, `MKL_THREADING_LAYER=GNU`, `PYTHONPATH` covers
`python/`, `third_party/litevloc_code/python` and `third_party/pose_estimation_models`,
`PYTHONUNBUFFERED=1`. `MERGE_CPU_LIST` (e.g. `0-11,16-31`) is passed to `taskset`; when it is
unset the backend detects the highest-clocked cores and avoids them, and an empty value
disables pinning. `CUDA_VISIBLE_DEVICES` is forwarded as is. The health endpoint reports the
CPU list actually in use.

**Fake pipeline for UI work.** `NAVMAP_CONSOLE_FAKE_PIPELINE=1` swaps the real pipeline for
`backend/navmap_console/jobs/fake_pipeline.py`, which copies each session into a step
directory and prints the same log anchors, so runs, jobs, progress, WebSocket streaming and
restart adoption can be exercised without a GPU. `NAVMAP_CONSOLE_FAKE_STEP_SECONDS` sets the
time per step (default 0.2) and `NAVMAP_CONSOLE_FAKE_FAIL_AT=<step>` makes that step crash.

**WebSocket.** One connection per tab at `/ws`. The client sends
`{"op": "sub" | "unsub", "topic": "..."}` with topics `jobs` (all job state changes),
`job:<id>` (state, progress and log lines of one job) and `run:<id>` (`run.state`,
`run.step_completed`). Every server message is
`{"topic", "type", "seq", "ts", "data"}`; subscribing to a `job:` topic first delivers a
`job.snapshot` with the job, the last 2000 log lines and `next_seq`, after which
`job.log` messages carry `{"id", "first_seq", "lines"}`. After a reconnect
`GET /api/jobs/<id>/log?after=<seq>` fills the gap. An unknown op is answered with a
`type: "error"` message and the connection stays open.

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
