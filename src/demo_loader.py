"""Demo Mode helpers: load previously generated analysis results from disk.

The dashboard normally builds a ``result`` dict live by running YOLO/Whisper,
then persists the full dict as ``outputs/metrics/<base>_metrics.json`` (see
``report_generator.save_all_exports``). For a smooth client demo we can skip the
long recompute and reload the latest saved result instantly.

Note: the persisted JSON contains every ``result`` key except ``exports`` (that
field is attached by ``app.py`` only after ``save_all_exports`` returns). We
therefore reconstruct the export download paths here from the sibling files that
share the JSON's base name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_output_dirs, project_root


def reconstruct_exports(json_path: Path, dirs: dict[str, Path]) -> dict[str, str]:
    """Rebuild the export map for a saved run from its sibling files on disk.

    Mirrors the key order produced by ``report_generator.save_all_exports`` and
    only includes files that actually exist.
    """
    base = json_path.name[: -len("_metrics.json")]
    metrics_dir = dirs["metrics"]
    reports_dir = dirs["reports"]

    candidates = [
        ("json", json_path),
        ("summary_csv", metrics_dir / f"{base}_summary.csv"),
        ("vision_timeseries_csv", metrics_dir / f"{base}_vision_timeseries.csv"),
        ("markdown_report", reports_dir / f"{base}_report.md"),
        ("html_report", reports_dir / f"{base}_report.html"),
        ("speaking_activity_csv", metrics_dir / f"{base}_speaking_activity.csv"),
    ]
    return {key: str(path) for key, path in candidates if path.exists()}


def _result_label(result: dict[str, Any]) -> str:
    media_name = result.get("media", {}).get("name", "clip")
    scenario_title = result.get("scenario", {}).get("title") or result.get("scenario_id", "")
    state = result.get("summary", {}).get("classroom_state", "")
    created = (result.get("created_at", "") or "").replace("T", " ")
    parts = [media_name]
    if scenario_title:
        parts.append(scenario_title)
    if state:
        parts.append(state)
    if created:
        parts.append(created)
    return " · ".join(parts)


def load_saved_results(root: Path | None = None) -> list[dict[str, Any]]:
    """Return saved analysis results, newest first.

    Each entry is the persisted ``result`` dict augmented with a reconstructed
    ``exports`` map and a human-readable ``demo_label``. Runs whose annotated
    video is missing are still returned (charts/KPIs remain valid); the app
    guards individual assets at render time.
    """
    base = root or project_root()
    dirs = ensure_output_dirs(base)
    json_paths = sorted(
        dirs["metrics"].glob("*_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for json_path in json_paths:
        try:
            result = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(result, dict) or "summary" not in result:
            continue
        result.setdefault("exports", {})
        if not result["exports"]:
            result["exports"] = reconstruct_exports(json_path, dirs)
        result["demo_label"] = _result_label(result)
        result["demo_source"] = str(json_path)
        results.append(result)
    return results
