import asyncio
from threading import Thread, Event
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 100
    temperature: float = 0.7

# 1. Create a custom stopping rule that PyTorch checks inside its C++ loop
class StreamCancelCriteria(StoppingCriteria):
    def __init__(self, stop_event: Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # If the stop event is set by FastAPI, returning True tells the C++ engine to STOP generating immediately
        return self.stop_event.is_set()

@router.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest, request: Request):
    device = request.app.state.device
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    
    # A thread-safe flag shared between FastAPI and the background C++ loop
    stop_event = Event()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": chat_request.message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 2. Bind the custom cancel criteria to our event flag
    cancel_criteria = StreamCancelCriteria(stop_event)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": chat_request.max_tokens,
        "temperature": chat_request.temperature,
        "do_sample": chat_request.temperature > 0,
        "stopping_criteria": StoppingCriteriaList([cancel_criteria]) # Inject it here!
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 3. Use an async generator to handle real-world connection checking
    async def generate_chunks():
        try:
            while thread.is_alive() or not streamer.empty():
                # Check if the user disconnected mid-stream
                if await request.is_disconnected():
                    print("User disconnected! Signal sent to stop GPU execution.")
                    stop_event.set() # Set the flag to halt the background C++ thread
                    break
                
                # Check the Python queue safely
                if not streamer.queue.empty():
                    text = streamer.queue.get()
                    if text is streamer.end_token:
                        break
                    yield text
                else:
                    # Give up the event loop briefly so other requests can run
                    await asyncio.sleep(0.01)
        finally:
            # Ensure the background thread is signaled to terminate if anything goes wrong
            stop_event.set()
            thread.join(timeout=1.0)

    return StreamingResponse(generate_chunks(), media_type="text/plain")
