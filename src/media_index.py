from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, file_size_mb, project_root, write_json


DEFAULT_SCENARIOS: dict[str, dict[str, Any]] = {
    "kids_walking": {
        "title": "Kids Walking / Transitions",
        "description": "Tracks students walking and transitioning inside classroom space.",
        "expected_metrics": ["person_count", "movement_energy", "transition_intensity"],
    },
    "kids_seated": {
        "title": "Kids Seated / Independent Learning",
        "description": "Monitors seated students, occupancy, and stillness as engagement proxies.",
        "expected_metrics": ["occupancy", "stationary_ratio", "engagement_proxy"],
    },
    "teacher_speaking": {
        "title": "Teacher Speaking",
        "description": "Analyzes teacher speech cadence, word count, and lesson transcript.",
        "expected_metrics": ["word_count", "words_per_minute", "speaking_time", "lesson_summary"],
    },
    "teacher_moving": {
        "title": "Teacher Moving",
        "description": "Tracks teacher circulation and classroom positioning using movement proxies.",
        "expected_metrics": ["teacher_movement_proxy", "movement_path", "zone_coverage"],
    },
    "aggregate_dynamics": {
        "title": "Aggregate Classroom Dynamics",
        "description": "Combines classroom occupancy, movement energy, and audio/activity signals.",
        "expected_metrics": ["classroom_state", "engagement_score", "activity_index"],
    },
}


DEFAULT_CONFIG: dict[str, Any] = {
    "scenarios": DEFAULT_SCENARIOS,
    "scan_dirs": [".", "data", "videos", "media", "assets", "clips", "input"],
    "processing_presets": {
        "Low": {"frame_stride": 10, "max_seconds": 60},
        "Medium": {"frame_stride": 5, "max_seconds": 90},
        "High": {"frame_stride": 2, "max_seconds": 120},
    },
    "manual_mappings": {},
}


@dataclass(frozen=True)
class MediaItem:
    id: str
    path: str
    name: str
    kind: str
    extension: str
    size_mb: float
    scenario_ids: list[str]
    scenario_title: str
    duration_seconds: float | None = None


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root() / "config" / "scenarios.yaml"
    config = DEFAULT_CONFIG.copy()
    if not path.exists():
        return config

    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return config

    merged = DEFAULT_CONFIG.copy()
    merged.update(loaded)
    merged["scenarios"] = {**DEFAULT_SCENARIOS, **loaded.get("scenarios", {})}
    merged.setdefault("manual_mappings", {})
    merged.setdefault("scan_dirs", DEFAULT_CONFIG["scan_dirs"])
    merged.setdefault("processing_presets", DEFAULT_CONFIG["processing_presets"])
    return merged


def scenario_options(config: dict[str, Any]) -> dict[str, str]:
    scenarios = config.get("scenarios", DEFAULT_SCENARIOS)
    return {key: value.get("title", key) for key, value in scenarios.items()}


def _normalise_mapping(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def _manual_scenarios(path: Path, root: Path, mappings: dict[str, Any]) -> list[str]:
    candidates = {path.name, path.stem}
    try:
        candidates.add(str(path.resolve().relative_to(root.resolve())).replace("\\", "/"))
    except Exception:
        pass

    lowered = {item.lower(): item for item in candidates}
    for key, value in mappings.items():
        if str(key).lower().replace("\\", "/") in lowered:
            return _normalise_mapping(value)
    return []


def _guess_scenarios(path: Path) -> list[str]:
    name = path.stem.lower().replace("-", "_")
    scenarios: list[str] = []

    if "task1" in name or "walking" in name or "transition" in name:
        scenarios.append("kids_walking")
    if "task3" in name or "speaking" in name or "speech" in name:
        scenarios.append("teacher_speaking")
    if "task5" in name or "aggregate" in name or "group" in name or "dynamics" in name:
        scenarios.append("aggregate_dynamics")
    if "task2_4" in name:
        scenarios.extend(["kids_seated", "teacher_moving"])
    if "seated" in name or "chair" in name or "independent" in name:
        scenarios.append("kids_seated")
    if ("teacher" in name and "moving" in name) or "circulation" in name:
        scenarios.append("teacher_moving")

    deduped: list[str] = []
    for scenario_id in scenarios:
        if scenario_id not in deduped:
            deduped.append(scenario_id)
    return deduped


def _candidate_roots(root: Path, scan_dirs: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in scan_dirs:
        path = (root / item).resolve()
        if path.exists() and path.is_dir() and path not in roots:
            roots.append(path)
    if root.resolve() not in roots:
        roots.insert(0, root.resolve())
    return roots


def _is_skipped(path: Path, root: Path) -> bool:
    skip_parts = {".git", ".venv", "venv", "__pycache__", "outputs", ".pytest_cache"}
    try:
        parts = {part.lower() for part in path.resolve().relative_to(root.resolve()).parts}
    except Exception:
        parts = {part.lower() for part in path.parts}
    return bool(parts.intersection(skip_parts))


def _video_duration(path: Path) -> float | None:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        if fps <= 0 or frames <= 0:
            return None
        return round(frames / fps, 2)
    except Exception:
        return None


def build_media_manifest(
    root: Path | None = None,
    config_path: Path | None = None,
    include_durations: bool = False,
) -> list[dict[str, Any]]:
    base = root or project_root()
    config = load_config(config_path)
    mappings = config.get("manual_mappings", {})
    scenarios = config.get("scenarios", DEFAULT_SCENARIOS)
    seen: set[Path] = set()
    items: list[MediaItem] = []

    for scan_root in _candidate_roots(base, config.get("scan_dirs", DEFAULT_CONFIG["scan_dirs"])):
        for path in scan_root.rglob("*"):
            if not path.is_file() or _is_skipped(path, base):
                continue
            suffix = path.suffix.lower()
            if suffix not in VIDEO_EXTENSIONS and suffix not in IMAGE_EXTENSIONS:
                continue

            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            kind = "video" if suffix in VIDEO_EXTENSIONS else "image"
            scenario_ids = _manual_scenarios(path, base, mappings) or _guess_scenarios(path)
            if not scenario_ids:
                scenario_ids = ["unmapped"]
            scenario_title = ", ".join(
                scenarios.get(item, {}).get("title", item.replace("_", " ").title())
                for item in scenario_ids
            )

            duration = _video_duration(path) if include_durations and kind == "video" else None
            items.append(
                MediaItem(
                    id=path.stem,
                    path=str(resolved),
                    name=path.name,
                    kind=kind,
                    extension=suffix,
                    size_mb=file_size_mb(path),
                    scenario_ids=scenario_ids,
                    scenario_title=scenario_title,
                    duration_seconds=duration,
                )
            )

    return [
        asdict(item)
        for item in sorted(items, key=lambda item: (item.scenario_ids[0], item.kind, item.name.lower()))
    ]


def save_media_manifest(manifest: list[dict[str, Any]], root: Path | None = None) -> Path:
    base = root or project_root()
    output_path = base / "outputs" / "metrics" / "media_manifest.json"
    write_json(output_path, manifest)
    return output_path


def media_for_scenario(manifest: list[dict[str, Any]], scenario_id: str) -> list[dict[str, Any]]:
    if scenario_id == "all":
        return manifest
    filtered = [item for item in manifest if scenario_id in item.get("scenario_ids", [])]
    return filtered or manifest
