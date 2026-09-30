from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from .config import setup
from .router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model, device, MODEL_NAME = setup()

    yield {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "MODEL_NAME": MODEL_NAME,
    }

    print("Cleaning up resources on shutdown...")


app = FastAPI(
    title="LLM Serving API",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(router)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model": request.state.MODEL_NAME,
        "device": request.state.device.type,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )