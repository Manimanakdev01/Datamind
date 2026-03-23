# -*- coding: utf-8 -*-
"""
yolo_api.py — FastAPI YOLO Detection Microservice
==================================================
Run locally:
    pip install -r requirements_yolo.txt
    uvicorn yolo_api:app --host 0.0.0.0 --port 8000

Deploy on Railway / Render / any VPS — separate from Streamlit Cloud.

Endpoints
---------
GET  /               health check
GET  /models         list available YOLO models
POST /detect         run YOLO on one image, return boxes + annotated image
POST /detect_batch   run YOLO on multiple images
"""

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("yolo_api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "DataMind AI — YOLO Detection API",
    description = "YOLO v8 object detection microservice for DataMind AI",
    version     = "1.0.0",
)

# Allow Streamlit Cloud to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Restrict to your Streamlit URL in production
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# ── YOLO model cache (load once, reuse) ───────────────────────────────────────
_MODEL_CACHE: dict = {}
SUPPORTED_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

def _load_model(model_name: str):
    """Load and cache a YOLO model. Downloads automatically on first use."""
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported model '{model_name}'. "
                                   f"Choose from: {SUPPORTED_MODELS}")
    if model_name not in _MODEL_CACHE:
        log.info(f"Loading YOLO model: {model_name}")
        from ultralytics import YOLO
        _MODEL_CACHE[model_name] = YOLO(model_name)
        log.info(f"Model {model_name} ready.")
    return _MODEL_CACHE[model_name]


def _run_detection(
    image_bytes: bytes,
    model_name:  str,
    confidence:  float,
    classes:     Optional[List[str]],
) -> dict:
    """
    Run YOLO detection on raw image bytes.
    Returns dict with boxes list + base64-encoded annotated PNG.
    """
    model  = _load_model(model_name)
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    det = model(pil_img, conf=confidence, verbose=False)[0]

    boxes = []
    for box in det.boxes:
        cls_id  = int(box.cls[0])
        cls_nm  = model.names[cls_id]
        conf_v  = float(box.conf[0])
        xyxy    = [round(float(v), 1) for v in box.xyxy[0]]

        if classes and cls_nm not in classes:
            continue

        boxes.append({
            "class":      cls_nm,
            "confidence": round(conf_v, 4),
            "x1": xyxy[0], "y1": xyxy[1],
            "x2": xyxy[2], "y2": xyxy[3],
        })

    # Annotated image → base64 PNG
    ann_arr  = det.plot()
    ann_img  = Image.fromarray(ann_arr)
    ann_buf  = io.BytesIO()
    ann_img.save(ann_buf, format="PNG")
    ann_b64  = base64.b64encode(ann_buf.getvalue()).decode("utf-8")

    return {
        "boxes":        boxes,
        "n_objects":    len(boxes),
        "classes":      list(set(b["class"] for b in boxes)),
        "ann_image_b64": ann_b64,        # base64 PNG of annotated image
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    """Health check — returns status and available models."""
    return {
        "status":           "ok",
        "service":          "DataMind AI YOLO API",
        "supported_models": SUPPORTED_MODELS,
        "cached_models":    list(_MODEL_CACHE.keys()),
    }


@app.get("/models", tags=["Models"])
def list_models():
    """List supported YOLO model names."""
    return {"models": SUPPORTED_MODELS}


@app.post("/detect", tags=["Detection"])
async def detect(
    file:       UploadFile = File(...,  description="Image file (JPG/PNG/WEBP)"),
    model:      str        = Form("yolov8n.pt", description="YOLO model name"),
    confidence: float      = Form(0.40,         description="Confidence threshold 0-1"),
    classes:    str        = Form("",           description="Comma-separated class filter (empty = all)"),
):
    """
    Run YOLO detection on a single image.

    Returns JSON:
    ```json
    {
        "filename":      "image.jpg",
        "n_objects":     3,
        "classes":       ["person", "car"],
        "boxes":         [{"class": "person", "confidence": 0.91, "x1": 10, ...}],
        "ann_image_b64": "<base64 PNG string>"
    }
    ```
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp",
                                  "image/jpg", "application/octet-stream"):
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {file.content_type}. "
                                   "Upload JPG, PNG, or WEBP.")

    t0          = time.time()
    image_bytes = await file.read()
    cls_filter  = [c.strip() for c in classes.split(",") if c.strip()] or None

    try:
        result = _run_detection(image_bytes, model, confidence, cls_filter)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    result["filename"]    = file.filename
    result["model"]       = model
    result["elapsed_s"]   = round(time.time() - t0, 3)
    return JSONResponse(content=result)


@app.post("/detect_batch", tags=["Detection"])
async def detect_batch(
    files:      List[UploadFile] = File(...),
    model:      str              = Form("yolov8n.pt"),
    confidence: float            = Form(0.40),
    classes:    str              = Form(""),
):
    """
    Run YOLO detection on multiple images in one request.
    Returns a list of per-image result dicts (same format as /detect).
    """
    cls_filter = [c.strip() for c in classes.split(",") if c.strip()] or None
    results    = []

    for file in files:
        image_bytes = await file.read()
        try:
            res = _run_detection(image_bytes, model, confidence, cls_filter)
            res["filename"] = file.filename
            res["error"]    = None
        except Exception as e:
            log.error(f"Batch detection error on {file.filename}: {e}")
            res = {"filename": file.filename, "error": str(e),
                   "boxes": [], "n_objects": 0, "classes": [], "ann_image_b64": ""}
        results.append(res)

    return JSONResponse(content={
        "model":    model,
        "n_images": len(files),
        "results":  results,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("yolo_api:app", host="0.0.0.0", port=port, reload=False)
