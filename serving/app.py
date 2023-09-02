from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

app = FastAPI()


class TypedRequest(BaseModel):
    strings: list[str] = Field(..., example=["Start", "Next"])


class TypedResponse(BaseModel):
    data: list[str]
    opt: Optional[dict[str, float]] = None


@app.get("/")
def read_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


@app.post("/predict", response_model=TypedResponse)
async def translate(request: TypedRequest) -> TypedResponse:
    logger.info(f"translate endpoint received request: {request}")
    request_strings = request.strings
    response = {"data": [f"{s}_response" for s in request_strings]}
    return TypedResponse(**response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
