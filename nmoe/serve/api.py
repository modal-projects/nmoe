# SPDX-License-Identifier: Apache-2.0
"""FastAPI server with OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nmoe.serve.orchestrator import AsyncOrchestrator, Orchestrator
from nmoe.serve.types import ForwardSpec, OutputMode, Request, SamplingParams


# OpenAI-compatible request/response models


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatCompletionRequest(BaseModel):
  model: str
  messages: list[ChatMessage]
  temperature: float = 1.0
  top_p: float = 1.0
  max_tokens: int = 256
  stream: bool = False
  seed: Optional[int] = None
  stop: Optional[list[str]] = None


class CompletionRequest(BaseModel):
  model: str
  prompt: str
  temperature: float = 1.0
  top_p: float = 1.0
  max_tokens: int = 256
  stream: bool = False
  seed: Optional[int] = None
  stop: Optional[list[str]] = None


class Usage(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class Choice(BaseModel):
  index: int
  message: Optional[ChatMessage] = None
  text: Optional[str] = None
  finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
  id: str
  object: str = "chat.completion"
  created: int
  model: str
  choices: list[Choice]
  usage: Usage


class CompletionResponse(BaseModel):
  id: str
  object: str = "text_completion"
  created: int
  model: str
  choices: list[Choice]
  usage: Usage


class DeltaMessage(BaseModel):
  role: Optional[str] = None
  content: Optional[str] = None


class StreamChoice(BaseModel):
  index: int
  delta: DeltaMessage
  finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
  id: str
  object: str = "chat.completion.chunk"
  created: int
  model: str
  choices: list[StreamChoice]


class HealthResponse(BaseModel):
  status: str


class ModelInfo(BaseModel):
  id: str
  object: str = "model"
  created: int
  owned_by: str = "nmoe"


class ModelsResponse(BaseModel):
  object: str = "list"
  data: list[ModelInfo]


def create_app(
  orchestrator: Orchestrator,
  tokenizer,  # HF tokenizer
  model_name: str = "deepseek-v3",
) -> FastAPI:
  """Create FastAPI app with OpenAI-compatible endpoints."""

  app = FastAPI(title="nmoe.serve", version="0.1.0")
  async_orch = AsyncOrchestrator(orchestrator)

  @app.on_event("startup")
  async def startup():
    await async_orch.start()

  @app.on_event("shutdown")
  async def shutdown():
    await async_orch.stop()

  @app.get("/health")
  async def health() -> HealthResponse:
    return HealthResponse(status="ok")

  @app.get("/v1/models")
  async def list_models() -> ModelsResponse:
    return ModelsResponse(
      data=[
        ModelInfo(
          id=model_name,
          created=int(time.time()),
        )
      ]
    )

  @app.post("/v1/chat/completions")
  async def chat_completions(request: ChatCompletionRequest):
    # Build prompt from messages
    prompt = _format_chat_messages(request.messages, tokenizer)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

    # Create sampling params
    sampling_params = SamplingParams(
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
      stop_sequences=request.stop or [],
    )

    # Create request
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
    )

    if request.stream:
      return StreamingResponse(
        _stream_chat_response(req, async_orch, tokenizer, model_name),
        media_type="text/event-stream",
      )

    # Non-streaming
    await async_orch.add_request(req)
    completed = await async_orch.wait_for_request(req, timeout=120.0)

    # Decode output
    output_text = tokenizer.decode(completed.output_ids, skip_special_tokens=True)

    return ChatCompletionResponse(
      id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
      created=int(time.time()),
      model=request.model,
      choices=[
        Choice(
          index=0,
          message=ChatMessage(role="assistant", content=output_text),
          finish_reason="stop" if completed.status.name == "FINISHED" else "length",
        )
      ],
      usage=Usage(
        prompt_tokens=len(input_ids),
        completion_tokens=len(completed.output_ids),
        total_tokens=len(input_ids) + len(completed.output_ids),
      ),
    )

  @app.post("/v1/completions")
  async def completions(request: CompletionRequest):
    # Tokenize
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")[0]

    # Create request
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
    )

    if request.stream:
      return StreamingResponse(
        _stream_completion_response(req, async_orch, tokenizer, model_name),
        media_type="text/event-stream",
      )

    # Non-streaming
    await async_orch.add_request(req)
    completed = await async_orch.wait_for_request(req, timeout=120.0)

    # Decode output
    output_text = tokenizer.decode(completed.output_ids, skip_special_tokens=True)

    return CompletionResponse(
      id=f"cmpl-{uuid.uuid4().hex[:8]}",
      created=int(time.time()),
      model=request.model,
      choices=[
        Choice(
          index=0,
          text=output_text,
          finish_reason="stop" if completed.status.name == "FINISHED" else "length",
        )
      ],
      usage=Usage(
        prompt_tokens=len(input_ids),
        completion_tokens=len(completed.output_ids),
        total_tokens=len(input_ids) + len(completed.output_ids),
      ),
    )

  return app


def _format_chat_messages(messages: list[ChatMessage], tokenizer) -> str:
  """Format chat messages into model prompt."""
  # Use tokenizer's chat template if available
  if hasattr(tokenizer, "apply_chat_template"):
    formatted = tokenizer.apply_chat_template(
      [{"role": m.role, "content": m.content} for m in messages],
      tokenize=False,
      add_generation_prompt=True,
    )
    return formatted

  # Fallback: simple format
  parts = []
  for msg in messages:
    if msg.role == "system":
      parts.append(f"System: {msg.content}\n")
    elif msg.role == "user":
      parts.append(f"User: {msg.content}\n")
    elif msg.role == "assistant":
      parts.append(f"Assistant: {msg.content}\n")
  parts.append("Assistant:")
  return "".join(parts)


async def _stream_chat_response(
  req: Request,
  async_orch: AsyncOrchestrator,
  tokenizer,
  model_name: str,
) -> AsyncIterator[str]:
  """Stream chat completion response."""
  import json

  req.return_queue = asyncio.Queue()
  await async_orch.add_request(req)

  request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
  created = int(time.time())

  # Initial response with role
  chunk = ChatCompletionStreamResponse(
    id=request_id,
    created=created,
    model=model_name,
    choices=[
      StreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
      )
    ],
  )
  yield f"data: {chunk.model_dump_json()}\n\n"

  # Stream tokens
  prev_len = 0
  while True:
    try:
      completed = await asyncio.wait_for(req.return_queue.get(), timeout=0.1)
      # Final chunk
      if len(req.output_ids) > prev_len:
        new_tokens = req.output_ids[prev_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        chunk = ChatCompletionStreamResponse(
          id=request_id,
          created=created,
          model=model_name,
          choices=[
            StreamChoice(
              index=0,
              delta=DeltaMessage(content=text),
            )
          ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

      # Done
      chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
          StreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop",
          )
        ],
      )
      yield f"data: {chunk.model_dump_json()}\n\n"
      yield "data: [DONE]\n\n"
      break

    except asyncio.TimeoutError:
      # Check for new tokens
      if len(req.output_ids) > prev_len:
        new_tokens = req.output_ids[prev_len:]
        prev_len = len(req.output_ids)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        chunk = ChatCompletionStreamResponse(
          id=request_id,
          created=created,
          model=model_name,
          choices=[
            StreamChoice(
              index=0,
              delta=DeltaMessage(content=text),
            )
          ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"


async def _stream_completion_response(
  req: Request,
  async_orch: AsyncOrchestrator,
  tokenizer,
  model_name: str,
) -> AsyncIterator[str]:
  """Stream completion response."""
  import json

  req.return_queue = asyncio.Queue()
  await async_orch.add_request(req)

  request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
  created = int(time.time())

  # Stream tokens
  prev_len = 0
  while True:
    try:
      completed = await asyncio.wait_for(req.return_queue.get(), timeout=0.1)
      # Final
      if len(req.output_ids) > prev_len:
        new_tokens = req.output_ids[prev_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = {
          "id": request_id,
          "object": "text_completion",
          "created": created,
          "model": model_name,
          "choices": [{"index": 0, "text": text, "finish_reason": None}],
        }
        yield f"data: {json.dumps(response)}\n\n"

      response = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
      }
      yield f"data: {json.dumps(response)}\n\n"
      yield "data: [DONE]\n\n"
      break

    except asyncio.TimeoutError:
      if len(req.output_ids) > prev_len:
        new_tokens = req.output_ids[prev_len:]
        prev_len = len(req.output_ids)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = {
          "id": request_id,
          "object": "text_completion",
          "created": created,
          "model": model_name,
          "choices": [{"index": 0, "text": text, "finish_reason": None}],
        }
        yield f"data: {json.dumps(response)}\n\n"
