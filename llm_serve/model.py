from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 100
    temperature: float = 0.7