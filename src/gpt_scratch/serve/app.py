"""FastAPI application: /health, /v1/generate, /v1/chat (JSON or SSE), and the web UI at /."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from gpt_scratch.serve.engine import Engine, PromptError
from gpt_scratch.serve.schemas import ChatRequest, GenerateRequest, Health, TextResponse
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


def create_app(run_dir: str | Path, device: str = "auto") -> FastAPI:
    engine = Engine(run_dir, device)
    app = FastAPI(title="gpt-scratch", docs_url=None, redoc_url=None)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", chat=engine.is_chat, params=engine.num_params, block_size=engine.block_size)

    @app.post("/v1/generate")
    def generate(req: GenerateRequest) -> Response:
        try:
            return _respond(engine.generate(req.prompt, req), req.stream)
        except PromptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/chat")
    def chat(req: ChatRequest) -> Response:
        try:
            return _respond(engine.chat(req.messages, req), req.stream)
        except PromptError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    return app
