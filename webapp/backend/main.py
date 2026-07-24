"""FastAPI app: JSON + SSE API over the acg instrument, and (in prod) the built SPA.

Run:  uvicorn webapp.backend.main:app --port 8100   (from the repo root)
The model server lives on :8000; this app uses :8100 to avoid the clash.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from typing import Any, Callable, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from . import corpus_store, families, history_store, paths, pipeline, presets, replay, serving
from .streaming import RUNS

app = FastAPI(title="ACG Experiment Dashboard", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Metadata endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def api_health(base_url: str | None = Query(None)):
    return presets.health(base_url)


@app.get("/api/models")
def api_models(base_url: str | None = Query(None)):
    return presets.models(base_url)


@app.get("/api/defaults")
def api_defaults():
    return presets.defaults()


@app.get("/api/prompts")
def api_prompts():
    return {"presets": presets.prompt_presets()}


# --------------------------------------------------------------------------- #
# Streaming runs (live) + replay — shared SSE plumbing
# --------------------------------------------------------------------------- #
def _sse_from_worker(run_id: str, runner: Callable[[Callable[[dict], None]], dict],
                     *, persist: bool) -> EventSourceResponse:
    """Register a queue, run `runner` in a thread, and stream its events as SSE.

    `runner(on_event)` produces the run; `on_event` and the tracing-layer streaming
    processor both push onto this run's queue (routed by run_id).
    """
    q = RUNS.register(run_id)
    done = threading.Event()

    def worker():
        def on_event(ev: dict) -> None:
            RUNS.emit(run_id, ev)
        try:
            runner(on_event)
        except Exception as e:  # already surfaced as an `error` event by the runner
            RUNS.emit(run_id, {"kind": "error", "run_id": run_id,
                               "error": f"{type(e).__name__}: {e}"})
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True, name=f"run-{run_id}").start()

    async def gen():
        saved = False
        try:
            while True:
                drained = False
                while True:
                    try:
                        ev = q.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    if ev.get("kind") == "run_finished" and persist and not saved:
                        try:
                            history_store.save(ev)
                            saved = True
                        except Exception:
                            pass
                    yield {"event": ev.get("kind", "message"), "data": json.dumps(ev, default=str)}
                if done.is_set() and q.empty():
                    break
                if not drained:
                    await asyncio.sleep(0.03)
        finally:
            RUNS.unregister(run_id)

    return EventSourceResponse(gen())


@app.post("/api/runs")
def api_run(req: dict = Body(...)):
    run_id = uuid.uuid4().hex[:12]
    return _sse_from_worker(
        run_id,
        lambda on_event: pipeline.run_live(run_id, req, on_event),
        persist=True,
    )


@app.get("/api/serving")
def api_serving():
    return serving.state()


@app.post("/api/serving/serve")
def api_serve(req: dict = Body(...)):
    model = req.get("model")
    if not model:
        raise HTTPException(400, "serve requires 'model'")
    run_id = uuid.uuid4().hex[:12]
    return _sse_from_worker(
        run_id,
        lambda on_event: serving.swap(model, on_event),
        persist=False,
    )


@app.post("/api/replay")
def api_replay(req: dict = Body(...)):
    file = req.get("file")
    if not file:
        raise HTTPException(400, "replay requires 'file'")
    run_id = uuid.uuid4().hex[:12]
    return _sse_from_worker(
        run_id,
        lambda on_event: replay.replay_run(
            file, req.get("trace_id"), on_event,
            speed=float(req.get("speed", 1.0)),
        ),
        persist=bool(req.get("save", False)),
    )


# --------------------------------------------------------------------------- #
# Traces (for the replay gallery)
# --------------------------------------------------------------------------- #
@app.get("/api/traces")
def api_traces():
    return {"traces": replay.list_traces()}


@app.get("/api/traces/runs")
def api_trace_runs(file: str = Query(...)):
    try:
        return {"runs": replay.trace_runs(file)}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.get("/api/families")
def api_families(files: str | None = Query(None)):
    """Per-family × per-backbone structural rollup (docs/12 view) from archived traces.

    `files` is an optional comma-separated list of trace files under traces/; defaults to
    the model-sweep traces (families_7b, families_fp8, families)."""
    flist = [f.strip() for f in files.split(",") if f.strip()] if files else None
    return families.rollup(flist)


# --------------------------------------------------------------------------- #
# Document manager (corpus CRUD)
# --------------------------------------------------------------------------- #
@app.get("/api/corpus")
def api_corpus_list(kind: str = "corpus"):
    return {"kind": kind, "docs": corpus_store.list_docs(kind)}


@app.get("/api/corpus/search")
def api_corpus_search(query: str, top_k: int = 3, noise: int = 0):
    return {"results": corpus_store.preview_search(query, top_k=top_k, noise=noise)}


@app.post("/api/corpus/reindex")
def api_corpus_reindex():
    return corpus_store.reindex()


@app.get("/api/corpus/{doc_id}")
def api_corpus_get(doc_id: str, kind: str = "corpus"):
    doc = corpus_store.get_doc(doc_id, kind)
    if doc is None:
        raise HTTPException(404, f"no document '{doc_id}'")
    return doc


@app.post("/api/corpus")
def api_corpus_create(doc: dict = Body(...), kind: str = "corpus"):
    try:
        return corpus_store.create_doc(doc, kind)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/corpus/{doc_id}")
def api_corpus_update(doc_id: str, doc: dict = Body(...), kind: str = "corpus"):
    try:
        return corpus_store.update_doc(doc_id, doc, kind)
    except KeyError:
        raise HTTPException(404, f"no document '{doc_id}'")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/corpus/{doc_id}")
def api_corpus_delete(doc_id: str, kind: str = "corpus"):
    try:
        corpus_store.delete_doc(doc_id, kind)
        return {"ok": True}
    except KeyError:
        raise HTTPException(404, f"no document '{doc_id}'")


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
@app.get("/api/history")
def api_history_list():
    return {"entries": history_store.list_entries()}


@app.get("/api/history/{entry_id}")
def api_history_get(entry_id: str):
    entry = history_store.get(entry_id)
    if entry is None:
        raise HTTPException(404, "no such history entry")
    return entry


@app.delete("/api/history/{entry_id}")
def api_history_delete(entry_id: str):
    return {"ok": history_store.delete(entry_id)}


# --------------------------------------------------------------------------- #
# Static SPA (production). Dev uses the Vite server on :5173 with a proxy.
# --------------------------------------------------------------------------- #
if paths.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=paths.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # Serve real files if present, else fall back to index.html for client routes.
        candidate = paths.FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        index = paths.FRONTEND_DIST / "index.html"
        if index.exists():
            # index.html references content-hashed asset bundles, so it must never be served
            # stale from cache or the browser keeps loading an old JS bundle (and the UI looks
            # unchanged after a redeploy). Force revalidation of the HTML entrypoint.
            return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return JSONResponse({"detail": "frontend not built"}, status_code=404)
