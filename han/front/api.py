"""Gather endpoints.

You can serve this app by `uvicorn han.front.api:app`.

"""
import os
import logging
import torch
import fastapi
import pydantic
from ..model import document as d
from ..model import sentence as s


__logger = logging.getLogger(__name__)

app = fastapi.FastAPI()

__logger.info("Loading a model.")

model: d.DocumentModel | s.SentenceModel = torch.load(os.environ["HAN_MODEL"])


class Texts(pydantic.BaseModel):
    """Request body of `classification`."""

    texts: list[str]


@app.post("/classification")
async def classification(texts: Texts):
    """Classify texts into classes."""
    print(texts)
    return {"message": "Hello World"}
