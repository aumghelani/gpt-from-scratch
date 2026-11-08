"""FastAPI application: /health, /v1/generate, /v1/chat (JSON or SSE), and the web UI at /."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from gpt_scratch.serve.engine import Engine, PromptError
from gpt_scratch.serve.schemas import ChatRequest, GenerateRequest, Health, ModelInfo, TextResponse
from gpt_scratch.serve.ui import INDEX_HTML


def _sse(pieces: Iterator[str]) -> Iterator[str]:
    for piece in pieces:
        yield f"data: {json.dumps({'token': piece})}\n\n"
    yield "data: [DONE]\n\n"


def _respond(pieces: Iterator[str], stream: bool) -> Response:
    if stream:
        return StreamingResponse(
            _sse(pieces), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )
    parts = list(pieces)
    return JSONResponse(TextResponse(text="".join(parts), tokens=len(parts)).model_dump())


def create_app(
    run_dirs: str | Path | Sequence[str | Path], device: str = "auto", checkpoint: str = "best"
) -> FastAPI:
    """Serve one or more trained runs; each is addressed by its directory name, the first is the default."""
    runs = [Path(run_dirs)] if isinstance(run_dirs, (str, Path)) else [Path(r) for r in run_dirs]
    if not runs:
        raise ValueError("at least one run directory is required")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate model names (run directories must have distinct names): {names}")
    engines = {run.name: Engine(run, device, checkpoint) for run in runs}
    default = names[0]
    app = FastAPI(title="gpt-scratch", docs_url=None, redoc_url=None)

    def pick(model: str | None) -> Engine:
        name = model or default
        if name not in engines:
            raise HTTPException(status_code=404, detail=f"unknown model {name!r}; available: {sorted(engines)}")
        return engines[name]

    @app.get("/health", response_model=Health)
    def health() -> Health:
        engine = engines[default]
        return Health(status="ok", model=default, chat=engine.is_chat, params=engine.num_params,
                      block_size=engine.block_size)

    @app.get("/v1/models", response_model=list[ModelInfo])
    def models() -> list[ModelInfo]:
        return [
            ModelInfo(id=name, chat=e.is_chat, params=e.num_params, block_size=e.block_size,
                      checkpoint=e.checkpoint_name)
            for name, e in engines.items()
        ]

    @app.post("/v1/generate")
    def generate(req: GenerateRequest) -> Response:
        try:
            return _respond(pick(req.model).generate(req.prompt, req), req.stream)
        except PromptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/chat")
    def chat(req: ChatRequest) -> Response:
        try:
            return _respond(pick(req.model).chat(req.messages, req), req.stream)
        except PromptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    return app
