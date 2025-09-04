import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR_PATH = Path("/home/debian/pkms/data")


class ProcessRequest(BaseModel):
    text: str
    timestamp: str


app = FastAPI()


@app.post("/process")
def process_string(request: ProcessRequest) -> str:
    file_name = f"{request.timestamp}.txt"
    (DATA_DIR_PATH / file_name).write_text(f"{request.text}\n")
    logger.info(f"Wrote \"{request.text}\" to \"{file_name}\"")
    return file_name


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

