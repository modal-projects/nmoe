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

from nmoe.serve.control_plane import (
  ControlPlane,
  OUTPUT_MODE_ID_TOKENS,
  finish_reason_id_to_str,
)
from nmoe.serve.orchestrator import AsyncOrchestrator, Orchestrator
from nmoe.serve.types import ForwardSpec, OutputMode, Request, RequestStatus, SamplingParams


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
  status: str  # "healthy", "degraded", "unhealthy"
  ready: bool = True
  running: bool = True
  queue_size: Optional[int] = None
  queue_capacity: Optional[int] = None
  details: Optional[dict] = None


class ErrorResponse(BaseModel):
  error: str
  code: str  # "rate_limited", "invalid_request", "server_error"
  limits: Optional[dict] = None


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
  *,
  control_plane: Optional[ControlPlane] = None,
) -> FastAPI:
  """Create FastAPI app with OpenAI-compatible endpoints."""

  app = FastAPI(title="nmoe.serve", version="0.1.0")
  async_orch = AsyncOrchestrator(orchestrator)
  proxy_reqs: dict[int, Request] = {}
  proxy_errors: dict[int, str] = {}

  def _apply_token_update(uid: int, token: int, done: bool, finish_reason_id: int) -> None:
    req = proxy_reqs.get(uid)
    if req is None:
      return
    if token >= 0:
      req.output_ids.append(int(token))
    if done:
      reason = finish_reason_id_to_str(finish_reason_id)
      req.finish_reason = reason
      if reason == "cancelled":
        req.status = RequestStatus.CANCELLED
      else:
        req.status = RequestStatus.FINISHED
      proxy_reqs.pop(uid, None)
      if reason != "error":
        proxy_errors.pop(uid, None)
    if req.return_queue is not None:
      req.return_queue.put_nowait(None)

  def _on_token_update(batch) -> None:
    # Called from receiver threads; hop onto the asyncio loop.
    loop = orchestrator._async_loop
    if loop is None:
      return
    uids = batch.uids.tolist()
    tokens = batch.tokens.tolist()
    flags = batch.uflags.tolist()

    def _apply_all() -> None:
      for uid, tok, fl in zip(uids, tokens, flags, strict=False):
        done = bool(int(fl) & 0x1)
        reason_id = int((int(fl) >> 1) & 0x7) if done else 0
        _apply_token_update(int(uid), int(tok), done, reason_id)

    loop.call_soon_threadsafe(_apply_all)

  def _on_error(uid: int, msg: str) -> None:
    loop = orchestrator._async_loop
    if loop is None:
      return

    def _apply_err() -> None:
      proxy_errors[int(uid)] = str(msg)
      _apply_token_update(int(uid), -1, True, 4)  # ERROR

    loop.call_soon_threadsafe(_apply_err)

  @app.on_event("startup")
  async def startup():
    await async_orch.start()
    if control_plane is not None:
      control_plane.start_rank0(
        on_token_update=_on_token_update,
        on_error=_on_error,
      )

  @app.on_event("shutdown")
  async def shutdown():
    await async_orch.stop()
    if control_plane is not None:
      control_plane.shutdown()

  @app.get("/health")
  async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
      - status: "healthy" (ready), "degraded" (starting), "unhealthy" (down)
      - HTTP 200 if healthy/degraded, 503 if unhealthy

    Use /health for liveness probes. Check `ready=true` for readiness probes.
    """
    health_status = async_orch.get_health_status()
    response = HealthResponse(
      status=health_status["status"],
      ready=health_status["ready"],
      running=health_status["running"],
      queue_size=health_status["queue_size"],
      queue_capacity=health_status["queue_capacity"],
      details=health_status["details"],
    )
    if health_status["status"] == "unhealthy":
      raise HTTPException(status_code=503, detail=response.model_dump())
    return response

  @app.get("/ready")
  async def ready():
    """Readiness probe - returns 200 only when ready to accept requests."""
    if not async_orch.is_ready:
      raise HTTPException(status_code=503, detail={"status": "not_ready"})
    return {"status": "ready"}

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

  @app.get("/v1/limits")
  async def get_limits():
    """Get server limits for client-side validation."""
    return async_orch.limits

  @app.post("/v1/chat/completions")
  async def chat_completions(request: ChatCompletionRequest):
    # Check if server is ready
    if not async_orch.is_ready:
      raise HTTPException(
        status_code=503,
        detail={"error": "Server not ready", "code": "server_not_ready"},
      )

    # Build prompt from messages
    prompt = _format_chat_messages(request.messages, tokenizer)

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    prompt_tokens = len(input_ids)

    # Validate bounds
    is_valid, error_msg = async_orch.validate_request_bounds(prompt_tokens, request.max_tokens)
    if not is_valid:
      raise HTTPException(
        status_code=400,
        detail={
          "error": error_msg,
          "code": "invalid_request",
          "limits": async_orch.limits,
        },
      )

    # Create sampling params
    sampling_params = SamplingParams(
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
      stop_sequences=request.stop or [],
    )

    # Create request (uid allocated on rank0; owner is uid % world_size)
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
    )

    # IMPORTANT: Set return_queue BEFORE enqueueing to avoid race condition.
    # If request completes before return_queue is set, _notify_request_done() would drop the signal.
    req.return_queue = asyncio.Queue()

    owner = int(req.uid) % int(orchestrator.world_size)
    if owner == 0:
      # Local ownership on rank0.
      if not async_orch.try_add_request(req):
        raise HTTPException(
          status_code=503,
          detail={
            "error": "Server overloaded, please retry",
            "code": "rate_limited",
            "queue_size": async_orch.get_health_status()["queue_size"],
            "queue_capacity": async_orch.limits["max_pending_requests"],
          },
        )
    else:
      if control_plane is None:
        raise HTTPException(status_code=500, detail={"error": "control_plane not configured"})
      proxy_reqs[int(req.uid)] = req
      control_plane.send_request_init(
        owner=owner,
        uid=int(req.uid),
        output_mode_id=OUTPUT_MODE_ID_TOKENS,
        topk=0,
        max_tokens=int(req.sampling_params.max_tokens),
        top_k=int(req.sampling_params.top_k),
        seed_or_minus1=int(req.sampling_params.seed) if req.sampling_params.seed is not None else -1,
        temperature=float(req.sampling_params.temperature),
        top_p=float(req.sampling_params.top_p),
        input_ids=req.input_ids,
      )

    if request.stream:
      # For streaming, request is already queued and return_queue is set
      return StreamingResponse(
        _stream_chat_response(
          req,
          async_orch,
          tokenizer,
          model_name,
          already_queued=True,
          owner=owner,
          control_plane=control_plane,
          proxy_errors=proxy_errors,
        ),
        media_type="text/event-stream",
      )

    # Non-streaming - request already in queue, return_queue already set
    try:
      completed = await async_orch.wait_for_request(req, timeout=120.0)
    except asyncio.TimeoutError:
      if control_plane is not None and owner != 0:
        control_plane.send_cancel(owner=owner, uid=int(req.uid))
      raise

    # Decode output
    if completed.finish_reason == "error":
      msg = proxy_errors.get(int(completed.uid), "unknown error")
      raise HTTPException(status_code=500, detail={"error": msg, "code": "internal_error"})
    output_text = tokenizer.decode(completed.output_ids, skip_special_tokens=True)

    return ChatCompletionResponse(
      id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
      created=int(time.time()),
      model=request.model,
      choices=[
        Choice(
          index=0,
          message=ChatMessage(role="assistant", content=output_text),
          finish_reason=completed.finish_reason or "stop",
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
    # Check if server is ready
    if not async_orch.is_ready:
      raise HTTPException(
        status_code=503,
        detail={"error": "Server not ready", "code": "server_not_ready"},
      )

    # Tokenize
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt")[0]
    prompt_tokens = len(input_ids)

    # Validate bounds
    is_valid, error_msg = async_orch.validate_request_bounds(prompt_tokens, request.max_tokens)
    if not is_valid:
      raise HTTPException(
        status_code=400,
        detail={
          "error": error_msg,
          "code": "invalid_request",
          "limits": async_orch.limits,
        },
      )

    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=request.temperature,
      top_p=request.top_p,
      max_tokens=request.max_tokens,
      seed=request.seed,
    )

    # IMPORTANT: Set return_queue BEFORE enqueueing to avoid race condition.
    req.return_queue = asyncio.Queue()

    owner = int(req.uid) % int(orchestrator.world_size)
    if owner == 0:
      if not async_orch.try_add_request(req):
        raise HTTPException(
          status_code=503,
          detail={
            "error": "Server overloaded, please retry",
            "code": "rate_limited",
            "queue_size": async_orch.get_health_status()["queue_size"],
            "queue_capacity": async_orch.limits["max_pending_requests"],
          },
        )
    else:
      if control_plane is None:
        raise HTTPException(status_code=500, detail={"error": "control_plane not configured"})
      proxy_reqs[int(req.uid)] = req
      control_plane.send_request_init(
        owner=owner,
        uid=int(req.uid),
        output_mode_id=OUTPUT_MODE_ID_TOKENS,
        topk=0,
        max_tokens=int(req.sampling_params.max_tokens),
        top_k=int(req.sampling_params.top_k),
        seed_or_minus1=int(req.sampling_params.seed) if req.sampling_params.seed is not None else -1,
        temperature=float(req.sampling_params.temperature),
        top_p=float(req.sampling_params.top_p),
        input_ids=req.input_ids,
      )

    if request.stream:
      return StreamingResponse(
        _stream_completion_response(
          req,
          async_orch,
          tokenizer,
          model_name,
          already_queued=True,
          owner=owner,
          control_plane=control_plane,
          proxy_errors=proxy_errors,
        ),
        media_type="text/event-stream",
      )

    # Non-streaming - request already in queue, return_queue already set
    try:
      completed = await async_orch.wait_for_request(req, timeout=120.0)
    except asyncio.TimeoutError:
      if control_plane is not None and owner != 0:
        control_plane.send_cancel(owner=owner, uid=int(req.uid))
      raise

    # Decode output
    if completed.finish_reason == "error":
      msg = proxy_errors.get(int(completed.uid), "unknown error")
      raise HTTPException(status_code=500, detail={"error": msg, "code": "internal_error"})
    output_text = tokenizer.decode(completed.output_ids, skip_special_tokens=True)

    return CompletionResponse(
      id=f"cmpl-{uuid.uuid4().hex[:8]}",
      created=int(time.time()),
      model=request.model,
      choices=[
        Choice(
          index=0,
          text=output_text,
          finish_reason=completed.finish_reason or "stop",
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
  already_queued: bool = False,
  *,
  owner: int = 0,
  control_plane: Optional[ControlPlane] = None,
  proxy_errors: Optional[dict[int, str]] = None,
) -> AsyncIterator[str]:
  """Stream chat completion response."""
  import json

  # Only set return_queue if not already set (avoid race condition)
  if req.return_queue is None:
    req.return_queue = asyncio.Queue()
  if not already_queued:
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

  prev_len = 0
  try:
    while True:
      await req.return_queue.get()
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

      if req.is_finished:
        chunk = ChatCompletionStreamResponse(
          id=request_id,
          created=created,
          model=model_name,
          choices=[
            StreamChoice(
              index=0,
              delta=DeltaMessage(),
              finish_reason=req.finish_reason or "stop",
            )
          ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        break
  except asyncio.CancelledError:
    # Client disconnected: propagate cancellation to the owner.
    req.cancel_flag = True
    if control_plane is not None and owner != 0:
      control_plane.send_cancel(owner=owner, uid=int(req.uid))
    raise


async def _stream_completion_response(
  req: Request,
  async_orch: AsyncOrchestrator,
  tokenizer,
  model_name: str,
  already_queued: bool = False,
  *,
  owner: int = 0,
  control_plane: Optional[ControlPlane] = None,
  proxy_errors: Optional[dict[int, str]] = None,
) -> AsyncIterator[str]:
  """Stream completion response."""
  import json

  # Only set return_queue if not already set (avoid race condition)
  if req.return_queue is None:
    req.return_queue = asyncio.Queue()
  if not already_queued:
    await async_orch.add_request(req)

  request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
  created = int(time.time())

  prev_len = 0
  try:
    while True:
      await req.return_queue.get()
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

      if req.is_finished:
        response = {
          "id": request_id,
          "object": "text_completion",
          "created": created,
          "model": model_name,
          "choices": [{"index": 0, "text": "", "finish_reason": req.finish_reason or "stop"}],
        }
        yield f"data: {json.dumps(response)}\n\n"
        yield "data: [DONE]\n\n"
        break
  except asyncio.CancelledError:
    req.cancel_flag = True
    if control_plane is not None and owner != 0:
      control_plane.send_cancel(owner=owner, uid=int(req.uid))
    raise
