"""Request/response models with strict bounds (all user input is validated here)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_TEXT_CHARS = 8000


class SamplingParams(BaseModel):
    max_new_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=1000)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    stream: bool = False


class GenerateRequest(SamplingParams):
    prompt: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ChatRequest(SamplingParams):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _ends_with_user_turn(self) -> ChatRequest:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return self


class TextResponse(BaseModel):
    text: str
    tokens: int


class Health(BaseModel):
    status: str
    chat: bool
    params: int
    block_size: int
