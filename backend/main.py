"""
AI Pothole Detection System - FastAPI Backend
================================================
A robust, production-oriented REST API around a YOLOv8 pothole-detection model.

Endpoints
---------
GET  /api/health   -> backend + model status
POST /api/detect    -> upload an image, get back detection counts, road score,
                       an annotated result image URL, and per-box detections
GET  /results/{f}   -> serve annotated result images
GET  /              -> serves the frontend (if built alongside this backend)
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("pothole-api")

# --------------------------------------------------------------------------- #
# Configuration (env-overridable so this is deployable as-is)
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE_DIR.parent / "frontend"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))

MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", 15))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
}

# Inference defaults, tuned in the original project for this dataset.
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", 0.15))
DEFAULT_IOU = float(os.getenv("DEFAULT_IOU", 0.20))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", 1280))

# Per-class acceptance thresholds applied *after* detection, mirroring the
# original app's hand-tuned filtering to cut down on false positives.
CLASS_CONF_THRESHOLDS = {
    "minor_pothole": float(os.getenv("CONF_MINOR", 0.55)),
    "medium_pothole": float(os.getenv("CONF_MEDIUM", 0.45)),
    "major_pothole": float(os.getenv("CONF_MAJOR", 0.40)),
}

CLASS_COLORS_BGR = {
    "minor_pothole": (86, 197, 34),    # green
    "medium_pothole": (14, 202, 250),  # amber
    "major_pothole": (60, 55, 239),    # red
}

# Keep the results folder from growing forever.
MAX_RESULT_FILES = int(os.getenv("MAX_RESULT_FILES", 200))

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]

# --------------------------------------------------------------------------- #
# Model loading (single instance, thread-safe access)
# --------------------------------------------------------------------------- #

model = None
model_error: Optional[str] = None
model_lock = Lock()


def load_model() -> None:
    """Load the YOLO model once at startup. Failures are captured, not fatal,
    so the API can still boot and report a clear 'model not loaded' status
    instead of crashing the whole process."""
    global model, model_error
    try:
        from ultralytics import YOLO

        weights_path = Path(MODEL_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at '{weights_path}'")

        loaded = YOLO(str(weights_path))
        model = loaded
        model_error = None
        logger.info("Model loaded from %s | classes: %s", weights_path, loaded.names)
    except Exception as exc:  # noqa: BLE001 - we want to capture *any* load failure
        model_error = str(exc)
        logger.error("Failed to load model: %s", exc)


def cleanup_old_results(keep_last: int = MAX_RESULT_FILES) -> None:
    """Prevent the results directory from growing without bound."""
    try:
        files = sorted(RESULTS_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - keep_last
        for f in files[:max(0, excess)]:
            f.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Result cleanup skipped: %s", exc)


# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="AI Pothole Detection API",
    description="Upload a road image and receive pothole severity detections.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    load_model()
    cleanup_old_results()


app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class Detection(BaseModel):
    class_name: str
    severity: str
    confidence: float
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")


class DetectResponse(BaseModel):
    id: str
    minor: int
    medium: int
    major: int
    total: int
    road_score: int
    road_condition: str
    image_url: str
    original_width: int
    original_height: int
    detections: List[Detection]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    classes: Optional[List[str]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SEVERITY_LABELS = {
    "minor_pothole": "Minor",
    "medium_pothole": "Medium",
    "major_pothole": "Major",
}


def compute_road_condition(score: int) -> str:
    if score <= 3:
        return "Good"
    if score <= 8:
        return "Moderate"
    return "Poor"


def validate_upload(file: UploadFile, raw: bytes) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'.",
        )

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB.",
        )


def decode_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # cheap corruption check
        img = Image.open(io.BytesIO(raw)).convert("RGB")  # re-open: verify() consumes the file
        return img
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the image. The file may be corrupted or not a real image.",
        ) from exc


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        classes=list(model.names.values()) if model is not None else None,
        error=model_error,
    )


@app.post(
    "/api/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["detection"],
)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    start = time.time()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded on the server: {model_error or 'unknown error'}",
        )

    raw = await file.read()
    validate_upload(file, raw)
    pil_img = decode_image(raw)

    width, height = pil_img.size
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    try:
        with model_lock:
            results = model(
                img_bgr,
                conf=DEFAULT_CONF,
                iou=DEFAULT_IOU,
                imgsz=DEFAULT_IMGSZ,
                verbose=False,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    minor = medium = major = 0
    detections: List[Detection] = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = model.names.get(cls_idx, str(cls_idx)) if isinstance(model.names, dict) else model.names[cls_idx]

            threshold = CLASS_CONF_THRESHOLDS.get(label, 0.25)
            if conf < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "minor_pothole":
                minor += 1
            elif label == "medium_pothole":
                medium += 1
            else:
                major += 1

            color = CLASS_COLORS_BGR.get(label, (255, 255, 255))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

            caption = f"{SEVERITY_LABELS.get(label, label)} {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_top = max(0, y1 - th - baseline - 8)
            cv2.rectangle(img_bgr, (x1, label_top), (x1 + tw + 10, y1), color, -1)
            cv2.putText(
                img_bgr,
                caption,
                (x1 + 5, y1 - 6 if y1 - 6 > th else label_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            detections.append(
                Detection(
                    class_name=label,
                    severity=SEVERITY_LABELS.get(label, label),
                    confidence=round(conf, 4),
                    box=[x1, y1, x2, y2],
                )
            )

    road_score = minor + (medium * 2) + (major * 3)
    road_condition = compute_road_condition(road_score)

    result_id = uuid.uuid4().hex[:12]
    result_filename = f"{result_id}.jpg"
    result_path = RESULTS_DIR / result_filename
    if not cv2.imwrite(str(result_path), img_bgr):
        raise HTTPException(status_code=500, detail="Failed to save the annotated result image.")

    cleanup_old_results()

    elapsed_ms = (time.time() - start) * 1000

    return DetectResponse(
        id=result_id,
        minor=minor,
        medium=medium,
        major=major,
        total=minor + medium + major,
        road_score=road_score,
        road_condition=road_condition,
        image_url=f"/results/{result_filename}",
        original_width=width,
        original_height=height,
        detections=detections,
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
    logger.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


# Mounted LAST and at the root so it never shadows the /api or /results
# routes above: Starlette matches routes in registration order, and this
# acts as a catch-all that also serves index.html for "/".
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("Frontend directory not found at %s (API-only mode)", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("RELOAD", "false").lower() == "true"),
    )
