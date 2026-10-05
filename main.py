import asyncio
import io
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.logger import log_main, logger
from app.schema import Classification, InferenceError, ModelStatus, NSFWResponse
from app.settings import SETTINGS

checker = None
checker_lock = threading.Lock()
model_status = ModelStatus.LOADING
load_error: str | None = None


def load_model() -> None:
    global checker, model_status, load_error
    try:
        from app.nsfw_check import NSFWCheck

        checker = NSFWCheck()
        model_status = ModelStatus.READY
    except Exception as e:
        logger.exception("Model load failed")
        load_error = repr(e)
        model_status = ModelStatus.FAILED


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.load_task = asyncio.create_task(asyncio.to_thread(load_model))
    yield


app = FastAPI(title="NSFW Detector", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    code = status.HTTP_200_OK if model_status == ModelStatus.READY else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=code,
        content={"status": model_status, "error": load_error},
    )


@app.post("/nsfw-check")
def nsfw_check(file: UploadFile) -> NSFWResponse:
    model = checker
    if model_status != ModelStatus.READY or model is None:
        raise HTTPException(status_code=503, detail=f"Model {model_status}")

    log_main("-------- Starting Task -------- ")
    task_id = str(uuid4())
    log_main(f"Task {task_id}: Received file {file.filename} for NSFW check.")

    log_main("Validating and Preprocessing the image...")

    data = file.file.read(SETTINGS.MAX_FILE_BYTES + 1)
    if len(data) > SETTINGS.MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File larger than {SETTINGS.MAX_FILE_BYTES // (1024 * 1024)} MB",
        )

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width > SETTINGS.MAX_SIDE_PX or height > SETTINGS.MAX_SIDE_PX:
            raise HTTPException(
                status_code=413,
                detail=f"Image {width}x{height} exceeds {SETTINGS.MAX_SIDE_PX}px per side",
            )
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File is not a valid image") from None

    if not checker_lock.acquire(timeout=SETTINGS.LOCK_WAIT_TIMEOUT_S):  # wait for the running task
        logger.warning(
            f"Task {task_id}: model busy for {SETTINGS.LOCK_WAIT_TIMEOUT_S}s, returning 503"
        )  # log so busy events can be counted
        raise HTTPException(status_code=503, detail="Model busy, route to another worker")  # backend re-routes on 503

    try:
        log_main("Running NSFW check...")
        result, duration = model.run_safety_check(image)  # inference on the device
    except InferenceError as e:  # inference failed
        logger.exception(f"Task {task_id}: {e}")  # log full error with task id
        raise HTTPException(status_code=500, detail=str(e)) from e  # error to the caller
    finally:
        checker_lock.release()

    log_main(f"Inference complete in {duration} seconds.")
    log_main(f"NSFW Results:\n{result.tsv()}")
    log_main("-------- Task Completed --------  \n")
    return NSFWResponse(
        task_id=task_id,
        is_nsfw=result.is_nsfw,
        classification=Classification(
            nsfw_score=result.nsfw_score,
            neutral=result.neutral,
            low=result.at_least("low"),
            medium=result.at_least("medium"),
            high=result.high,
        ),
    )


if __name__ == "__main__":
    uvicorn.run(app, host=SETTINGS.UVICORN_HOST, port=SETTINGS.UVICORN_PORT, access_log=False)
