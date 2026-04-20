from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from pathlib import Path
from typing import Any
import tempfile
import shutil
import os
import warnings
import asyncio

# Suppress warnings
warnings.filterwarnings("ignore")

# Keep CPU thread usage conservative for container stability.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Use conservative CPU kernel selection for broader hosted-CPU compatibility.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

try:
    import torch
    from fastai.vision.all import load_learner, PILImage
    from PIL import Image, UnidentifiedImageError
except ImportError:
    raise RuntimeError(
        "FastAI is not installed. Please install fastai and torch.")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="Pneumonia Detection API")

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Find the model in the current directory or parent directories
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

possible_paths = [
    current_dir / "export.pkl",
    project_root / "model" / "export.pkl",
    project_root / "artifacts" / "export.pkl",
]

model_path = None
for p in possible_paths:
    if p.exists():
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError("Could not find export.pkl.")

print(f"Loading model from: {model_path}")
learn = load_learner(model_path)
learn.model.eval()

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # MKLDNN can trigger native crashes on some constrained CPUs; disable for stability.
    torch.backends.mkldnn.enabled = False
except RuntimeError:
    # Thread settings can only be set once per process.
    pass

predict_lock = asyncio.Lock()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Pneumonia Detection API is running",
        "docs": "/docs",
        "predict": ["/predict"],
        "accepted_file_fields": ["file"],
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile | None = File(default=None)):
    incoming_file: Any = file
    if incoming_file is None:
        form = await request.form()
        for key in ("file", "image", "upload"):
            candidate = form.get(key)
            if isinstance(candidate, StarletteUploadFile):
                incoming_file = candidate
                break

    if incoming_file is None or not incoming_file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Use multipart field name: file (or image/upload).",
        )

    suffix = Path(incoming_file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(incoming_file.file, tmp)
        tmp_path = Path(tmp.name)

    normalized_path = tmp_path.with_suffix(".normalized.jpg")

    try:
        file_size = tmp_path.stat().st_size
        if file_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed size is {MAX_UPLOAD_MB} MB.",
            )

        print(
            f"[predict] file={incoming_file.filename} bytes={file_size}", flush=True)

        # Normalize format/mode before inference to avoid decoder/backend edge-case crashes.
        try:
            with Image.open(tmp_path) as raw_img:
                raw_img.load()
                rgb_img = raw_img.convert("RGB")
                if max(rgb_img.size) > 2048:
                    rgb_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                rgb_img.save(normalized_path, format="JPEG", quality=95)
        except UnidentifiedImageError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {e}")

        img = PILImage.create(normalized_path)
        # FastAI progress bars can break in some hosted environments; disable per-call.
        async with predict_lock:
            with learn.no_bar():
                with torch.inference_mode():
                    pred_label, _, probabilities = learn.predict(img)

        vocab = [str(label).strip() for label in learn.dls.vocab]
        class_probs = {
            class_name: float(prob)
            for class_name, prob in zip(vocab, probabilities.tolist())
        }

        def get_prob(*aliases: str) -> float:
            alias_set = {a.lower() for a in aliases}
            for key, value in class_probs.items():
                if key.lower() in alias_set:
                    return value
            return 0.0

        normal_prob = get_prob("normal")
        pneumonia_prob = get_prob("pneumonia")

        # Use explicit non-chest-Xray classes when available.
        other_prob = get_prob(
            "other",
            "not a chest-xray",
            "not a chest xray",
            "non-chest-xray",
            "non chest xray",
            "not_chest_xray",
        )

        # If model provides chest-xray probability, invert it to estimate "other".
        if other_prob == 0.0:
            chest_xray_prob = get_prob(
                "chest x-ray image",
                "chest xray image",
                "chest_xray",
            )
            if chest_xray_prob > 0.0:
                other_prob = max(0.0, 1.0 - chest_xray_prob)

        label_scores = {
            "normal": normal_prob,
            "pneumonia": pneumonia_prob,
            "other, not a chest-xray": other_prob,
        }

        result = max(label_scores.items(), key=lambda item: item[1])[0]

        # For binary models with no "other" class, treat low-confidence cases as non-chest-Xray.
        if other_prob == 0.0 and max(normal_prob, pneumonia_prob) < 0.60:
            result = "other, not a chest-xray"

        # Keep API contract deterministic even if vocab names are unusual.
        pred_text = str(pred_label).strip().lower()
        if pred_text == "normal":
            result = "normal"
        elif pred_text == "pneumonia":
            result = "pneumonia"

        print(f"[predict] result={result}", flush=True)
        return PlainTextResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict] error={e}", flush=True)
        raise HTTPException(
            status_code=500, detail=f"Error predicting: {str(e)}")
    finally:
        if normalized_path.exists():
            normalized_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
