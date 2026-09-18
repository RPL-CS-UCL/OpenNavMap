# app/

Two independent applications live here.

| Directory | What it is | Status |
|---|---|---|
| `litevloc_altas/` | Legacy Gradio demo: upload one image, run LiteVLoc visual localization against a pre-built atlas. Entry `litevloc_altas_app.py`, launch scripts in `scripts/`. | Frozen. Its `sys.path` entries predate the current repository layout; kept for reference only. |
| `navmap_console/` | Web control console for multi-session mapping: register/upload sessions, launch and monitor incremental map merges, inspect intermediate results interactively, evaluate, and export LiteVLoc navigation maps. FastAPI backend + React frontend. | Active. See `navmap_console/README.md`. |

Quick start for the console:

```bash
cd app/navmap_console
bash scripts/build.sh     # one-time: install frontend deps and build static files
bash scripts/serve.sh     # serve on http://0.0.0.0:8765
```
