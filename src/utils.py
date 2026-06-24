from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_output_dirs(root: Path | None = None) -> dict[str, Path]:
    base = root or project_root()
    dirs = {
        "outputs": base / "outputs",
        "annotated_videos": base / "outputs" / "annotated_videos",
        "reports": base / "outputs" / "reports",
        "metrics": base / "outputs" / "metrics",
        "audio": base / "outputs" / "metrics" / "audio",
        "cache": base / "outputs" / "cache",
        "ultralytics": base / "outputs" / "cache" / "ultralytics",
        "matplotlib": base / "outputs" / "cache" / "matplotlib",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_local_runtime_dirs(root: Path | None = None) -> dict[str, Path]:
    dirs = ensure_output_dirs(root)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(dirs["ultralytics"]))
    os.environ.setdefault("MPLCONFIGDIR", str(dirs["matplotlib"]))
    return dirs


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "item"


def file_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except OSError:
        return 0.0


def get_device_info() -> dict[str, Any]:
    configure_local_runtime_dirs()
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_available": False,
        "cuda_available": False,
        "torch_cuda_version": None,
        "device": "cpu",
        "gpu_name": None,
        "torch_version": None,
        "ultralytics_available": False,
        "ultralytics_version": None,
    }
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["torch_error"] = str(exc)
    try:
        import ultralytics

        info["ultralytics_available"] = True
        info["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["ultralytics_error"] = str(exc)
    return info


def print_runtime_banner() -> dict[str, Any]:
    info = get_device_info()
    print("[Classroom Intelligence Demo] Runtime status")
    print(f"  Python: {info.get('python_version')}")
    print(f"  torch: {info.get('torch_version') or 'not installed'}")
    print(f"  torch.cuda.is_available(): {info.get('cuda_available')}")
    print(f"  torch CUDA version: {info.get('torch_cuda_version') or 'None'}")
    print(f"  GPU name: {info.get('gpu_name') or 'CPU only'}")
    print(f"  ultralytics: {info.get('ultralytics_version') or 'not installed'}")
    if not info.get("cuda_available"):
        print("  WARNING: GPU not active. Install CUDA-enabled PyTorch.")
    return info


def find_ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def relative_to_root(path: str | Path, root: Path | None = None) -> str:
    root_path = root or project_root()
    try:
        return str(Path(path).resolve().relative_to(root_path.resolve()))
    except Exception:
        return str(path)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
