import time
import torch 
from fastapi import APIRouter, HTTPException, Request
from .model import ChatRequest

from threading import Thread

from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

router = APIRouter()

@router.post("/chat")
def chat(chat_request: ChatRequest, request: Request):
    device = request.state.device
    tokenizer = request.state.tokenizer
    model = request.state.model

    if chat_request.max_tokens < 1 or chat_request.max_tokens > 2048:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be between 1 and 2048",
        )

    if chat_request.temperature < 0:
        raise HTTPException(
            status_code=400,
            detail="temperature must be >= 0",
        )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": chat_request.message,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    start = time.perf_counter()

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature,
            do_sample=chat_request.temperature > 0,
        )

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    generated_ids = output_ids[0][
        inputs["input_ids"].shape[-1]:
    ]

    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return {
        "response": response_text,
        "latency_seconds": elapsed,
        "generated_tokens": len(generated_ids),
    }

@router.post("/chat/stream")
def chat_stream(chat_request: ChatRequest, request: Request):
    device = request.state.device
    tokenizer = request.state.tokenizer
    model = request.state.model
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": chat_request.message,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": chat_request.max_tokens,
        "temperature": chat_request.temperature,
        "do_sample": chat_request.temperature > 0,
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )

    thread.start()

    def generate_chunks():
        for text in streamer:
            yield text

    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain",
    )