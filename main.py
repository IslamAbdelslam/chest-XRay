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
import logging
import multiprocessing as mp
import time
import traceback

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
PREDICT_TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "50"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1024"))
CONFIGURED_INFERENCE_START_METHOD = os.getenv("INFERENCE_START_METHOD")
SAFE_INFERENCE_START_METHOD = os.getenv("SAFE_INFERENCE_START_METHOD", "spawn")
INFERENCE_CRASH_THRESHOLD = int(os.getenv("INFERENCE_CRASH_THRESHOLD", "2"))

logger = logging.getLogger("uvicorn.error")

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

learn = None
model_load_error = None
active_inference_start_method = SAFE_INFERENCE_START_METHOD
consecutive_inference_crashes = 0
last_prediction_vocab: list[str] = []


def load_model() -> None:
    global model_load_error

    if model_path is None:
        model_load_error = "Could not find export.pkl."
        logger.error(model_load_error)
        return

    model_load_error = None


@app.on_event("startup")
async def startup_event() -> None:
    load_model()

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
        "predict_timeout_seconds": PREDICT_TIMEOUT_SECONDS,
        "max_image_dim": MAX_IMAGE_DIM,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model_load_error is None,
        "model_error": model_load_error,
    }


@app.get("/diag")
async def diagnostics():
    return {
        "status": "ok",
        "model_loaded": model_load_error is None,
        "model_error": model_load_error,
        "model_path": str(model_path) if model_path is not None else None,
        "vocab": last_prediction_vocab,
        "settings": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "predict_timeout_seconds": PREDICT_TIMEOUT_SECONDS,
            "max_image_dim": MAX_IMAGE_DIM,
            "configured_inference_start_method": CONFIGURED_INFERENCE_START_METHOD,
            "inference_start_method": active_inference_start_method,
            "safe_inference_start_method": SAFE_INFERENCE_START_METHOD,
            "inference_crash_threshold": INFERENCE_CRASH_THRESHOLD,
            "consecutive_inference_crashes": consecutive_inference_crashes,
        },
        "runtime": {
            "pythonunbuffered": os.getenv("PYTHONUNBUFFERED"),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
            "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
            "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
            "aten_cpu_capability": os.getenv("ATEN_CPU_CAPABILITY"),
        },
        "lock": {
            "predict_lock_locked": predict_lock.locked(),
        },
        "versions": {
            "torch": getattr(torch, "__version__", None),
        },
    }


def _predict_from_path(image_path: Path):
    # Run all model work in one sync function so it can be moved to a worker thread.
    if learn is None:
        raise RuntimeError(model_load_error or "Model is not loaded")

    img = PILImage.create(image_path)
    with learn.no_bar():
        with torch.inference_mode():
            return learn.predict(img)


class InferenceSubprocessCrash(RuntimeError):
    def __init__(self, exit_code: int | None):
        self.exit_code = exit_code
        super().__init__(
            f"Inference subprocess crashed (exit code {exit_code}).")


def _predict_subprocess_worker(image_path: str, model_path_str: str | None, conn) -> None:
    try:
        local_learn = learn
        if local_learn is None:
            if not model_path_str:
                raise RuntimeError(
                    "Model path is missing in subprocess worker")
            local_learn = load_learner(Path(model_path_str))
            local_learn.model.eval()

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.backends.mkldnn.enabled = False
        except RuntimeError:
            pass

        img = PILImage.create(Path(image_path))
        with local_learn.no_bar():
            with torch.inference_mode():
                pred_label, _, probabilities = local_learn.predict(img)

        payload = {
            "ok": True,
            "pred_label": str(pred_label),
            "probabilities": probabilities.tolist(),
            "vocab": [str(label).strip() for label in local_learn.dls.vocab],
        }
        conn.send(payload)
    except Exception as exc:
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_repr": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


def _predict_via_subprocess(image_path: Path, timeout_seconds: float, start_method: str):
    ctx = mp.get_context(start_method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_predict_subprocess_worker,
        args=(str(image_path), str(model_path)
              if model_path is not None else None, child_conn),
        daemon=True,
    )

    try:
        proc.start()
        child_conn.close()
        deadline = time.monotonic() + timeout_seconds

        while True:
            if parent_conn.poll(0.2):
                try:
                    payload = parent_conn.recv()
                except EOFError:
                    if not proc.is_alive():
                        raise InferenceSubprocessCrash(proc.exitcode)
                    raise RuntimeError(
                        "Inference subprocess closed its pipe without returning a result."
                    )
                proc.join(timeout=1)
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"Unexpected inference payload type: {type(payload).__name__}"
                    )
                if not payload.get("ok"):
                    error_type = payload.get(
                        "error_type") or "InferenceWorkerError"
                    error_message = (
                        payload.get("error_message")
                        or payload.get("error_repr")
                        or "Unknown inference error"
                    )
                    traceback_text = payload.get("traceback")
                    if traceback_text:
                        logger.error(
                            "Inference worker traceback:\n%s", traceback_text)
                    raise RuntimeError(f"{error_type}: {error_message}")
                return payload

            if not proc.is_alive():
                raise InferenceSubprocessCrash(proc.exitcode)

            if time.monotonic() >= deadline:
                raise TimeoutError("Inference subprocess timed out")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
        parent_conn.close()


def _record_inference_success() -> None:
    global consecutive_inference_crashes
    consecutive_inference_crashes = 0


def _record_inference_crash() -> bool:
    global consecutive_inference_crashes, active_inference_start_method
    consecutive_inference_crashes += 1
    should_switch = (
        active_inference_start_method != SAFE_INFERENCE_START_METHOD
        and consecutive_inference_crashes >= INFERENCE_CRASH_THRESHOLD
    )
    if should_switch:
        logger.warning(
            "Switching inference subprocess mode from %s to %s after %d consecutive crashes",
            active_inference_start_method,
            SAFE_INFERENCE_START_METHOD,
            consecutive_inference_crashes,
        )
        active_inference_start_method = SAFE_INFERENCE_START_METHOD
        consecutive_inference_crashes = 0
        return True
    return False


@app.post("/predict")
async def predict(request: Request, file: UploadFile | None = File(default=None)):
    load_model()
    if model_load_error is not None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model is not available.",
        )

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
                if max(rgb_img.size) > MAX_IMAGE_DIM:
                    rgb_img.thumbnail(
                        (MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
                rgb_img.save(normalized_path, format="JPEG", quality=95)
        except UnidentifiedImageError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {e}")

        # FastAI progress bars can break in some hosted environments; disable per-call.
        async with predict_lock:
            try:
                prediction = await asyncio.to_thread(
                    _predict_via_subprocess,
                    normalized_path,
                    PREDICT_TIMEOUT_SECONDS,
                    active_inference_start_method,
                )
            except TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Prediction timed out before platform edge timeout. "
                        "Try a smaller image or increase resources."
                    ),
                )
            except InferenceSubprocessCrash as exc:
                switched = _record_inference_crash()
                detail = f"Inference worker crashed (exit code {exc.exit_code})."
                if switched:
                    detail += " Automatically switched to safer inference mode; retry the request."
                raise HTTPException(status_code=503, detail=detail)

        _record_inference_success()

        pred_label = prediction["pred_label"]
        probabilities = prediction["probabilities"]
        vocab = prediction["vocab"]
        global last_prediction_vocab
        last_prediction_vocab = vocab
        class_probs = {
            class_name: float(prob)
            for class_name, prob in zip(vocab, probabilities)
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
                "other",
                "Other",
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
        print(f"[predict] error={type(e).__name__}: {e!r}", flush=True)
        error_message = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting: {type(e).__name__}: {error_message}",
        )
    finally:
        if normalized_path.exists():
            normalized_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
