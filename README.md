# RoadScan — AI Pothole Detection System

A YOLOv8-based pothole detector with a FastAPI backend and a from-scratch
modern frontend. Upload a road photo and get back severity-graded bounding
boxes, per-class counts, and a road condition score.

This is a rebuild of the original Flask + Jinja prototype: same trained
model and detection logic, but with a real REST API, input validation,
error handling, and a redesigned UI (no more hardcoded `D:/...` Windows
paths, no server crashes on bad uploads, no page reloads on every submit).

## Project layout

```
pothole-detection-system/
├── backend/
│   ├── main.py            FastAPI app (see below)
│   ├── requirements.txt
│   ├── models/best.pt     trained YOLOv8 weights (minor/medium/major pothole)
│   └── results/           annotated output images (auto-created, auto-pruned)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train.py                original training script (reference only)
├── split_dataset.py
├── convert.py
└── README.md
```

## Running it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python main.py
```

The backend serves **both** the API and the frontend on the same origin, so
just open:

```
http://localhost:8000
```

No separate frontend server or build step needed — `main.py` mounts the
`frontend/` folder directly.

### Configuration

Everything is overridable via environment variables, so this runs the same
in dev and in a container:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `backend/models/best.pt` | path to YOLO weights |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | bind address |
| `MAX_FILE_SIZE_MB` | `15` | max upload size |
| `DEFAULT_CONF` / `DEFAULT_IOU` / `DEFAULT_IMGSZ` | `0.15` / `0.20` / `1280` | YOLO inference params |
| `CONF_MINOR` / `CONF_MEDIUM` / `CONF_MAJOR` | `0.55` / `0.45` / `0.40` | per-class acceptance thresholds |
| `ALLOWED_ORIGINS` | `*` | comma-separated CORS origins for production |
| `MAX_RESULT_FILES` | `200` | oldest annotated images beyond this are pruned |

## API

**`GET /api/health`**
```json
{ "status": "ok", "model_loaded": true, "model_path": "...", "classes": ["minor_pothole", "medium_pothole", "major_pothole"], "error": null }
```

**`POST /api/detect`** — multipart form, field name `file`
```json
{
  "id": "b059e51e7651",
  "minor": 2, "medium": 1, "major": 0, "total": 3,
  "road_score": 4, "road_condition": "Moderate",
  "image_url": "/results/b059e51e7651.jpg",
  "original_width": 800, "original_height": 600,
  "detections": [
    { "class_name": "minor_pothole", "severity": "Minor", "confidence": 0.71, "box": [120, 80, 210, 160] }
  ],
  "processing_time_ms": 812.4
}
```

Validation errors return `400` (bad file type / corrupt image / empty file),
`413` (too large), and `503` if the model failed to load — all with a plain
`{"detail": "..."}` message instead of a raw traceback.

## What changed from the original prototype

- **Backend**: Flask + `render_template` → FastAPI with a real JSON API,
  Pydantic response models, per-request validation (file type, content
  type, size, corruption), thread-safe model access, a global exception
  handler so a bad request can't 500 the whole page, auto-pruning of the
  results folder, and every path/threshold moved out of hardcoded strings
  into environment variables (the old code pointed at a literal
  `D:/pothole-detection-system-using-convolution-neural-networks-master/...`
  path that only existed on the original author's machine).
- **Frontend**: rebuilt from scratch as a single-page app — drag-and-drop
  upload with client-side validation, a scanning animation while the model
  runs, an animated road-condition gauge, a sortable detection log with
  per-box confidence bars, a backend status indicator, and toast
  notifications for errors instead of failed page reloads.
- **Model & thresholds**: unchanged — same `best.pt`, same class-wise
  confidence thresholds and road-score formula as the original `app.py`.

## Retraining

`train.py`, `split_dataset.py`, and `convert.py` are kept at the project
root for reference if you want to retrain on your own dataset. They're
unchanged from the original project and aren't part of the running app.
