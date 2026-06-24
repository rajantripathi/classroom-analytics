from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from .utils import ensure_output_dirs, safe_slug, timestamp_slug, write_json


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _markdown_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def generate_markdown_report(result: dict[str, Any]) -> str:
    media = result.get("media", {})
    scenario = result.get("scenario", {})
    summary = result.get("summary", {})
    vision = result.get("vision", {})
    audio = result.get("audio", {})

    kpis = [
        ("Average occupancy", summary.get("average_occupancy", 0)),
        ("Peak occupancy", summary.get("peak_occupancy", 0)),
        ("Movement energy", summary.get("movement_energy", 0)),
        ("Stationary/seated proxy", f"{float(summary.get('stationary_ratio', 0)):.0%}"),
        ("Transition intensity", summary.get("transition_intensity", 0)),
        ("Speaking pace", f"{summary.get('speaking_pace_wpm', 0)} WPM"),
        ("Word count", summary.get("word_count", 0)),
        ("Classroom state", summary.get("classroom_state", "Low Activity")),
    ]

    transcript = audio.get("transcript") or ""
    transcript_excerpt = transcript[:1200] + ("..." if len(transcript) > 1200 else "")
    warnings = list(vision.get("warnings") or []) + list(audio.get("warnings") or [])
    warning_block = "\n".join(f"- {item}" for item in warnings) if warnings else "- None"

    return f"""# Classroom Intelligence Demo Report

## Clip

- Media: {media.get("name", "Unknown")}
- Scenario: {scenario.get("title", "Unknown")}
- Generated: {result.get("created_at", "")}
- Detector: {vision.get("detector", "n/a")}
- Device: {vision.get("device", "cpu")}
- Analysis window: {vision.get("analysis_duration_seconds", 0)} seconds

## KPI Summary

{_markdown_table(kpis)}

## AI Insight Summary

{summary.get("ai_insight_summary", "")}

## Audio Summary

- Status: {audio.get("status", "not_run")}
- Lesson summary: {audio.get("lesson_summary") or "No transcript summary available."}

## Transcript Excerpt

{transcript_excerpt or "No transcript available."}

## Warnings

{warning_block}

## Privacy Note

This demo uses aggregate classroom analytics only. It does not perform face recognition,
student identification, identity matching, or emotion recognition.
"""


def generate_html_report(markdown_report: str, result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    title = html.escape(f"Classroom Intelligence Demo - {result.get('media', {}).get('name', 'Report')}")
    insight = html.escape(summary.get("ai_insight_summary", ""))
    escaped_markdown = html.escape(markdown_report)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Aptos, Segoe UI, sans-serif;
      color: #14202b;
      background: #f4f7f8;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    header {{
      border-bottom: 4px solid #1f7a6d;
      padding-bottom: 20px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .insight {{
      background: #e6f2ef;
      border-left: 5px solid #1f7a6d;
      padding: 18px;
      margin: 20px 0;
      font-size: 18px;
      line-height: 1.45;
    }}
    pre {{
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid #d8e1e5;
      border-radius: 8px;
      padding: 22px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Classroom Intelligence Demo Report</h1>
      <div>Anonymous aggregate analytics for classroom-level review</div>
    </header>
    <section class="insight">{insight}</section>
    <pre>{escaped_markdown}</pre>
  </main>
</body>
</html>
"""


def save_all_exports(result: dict[str, Any], output_root: Path | None = None) -> dict[str, str]:
    dirs = ensure_output_dirs(output_root)
    media_name = safe_slug(Path(result.get("media", {}).get("name", "clip")).stem)
    stamp = timestamp_slug()
    base_name = f"{media_name}_{stamp}"

    json_path = dirs["metrics"] / f"{base_name}_metrics.json"
    summary_csv_path = dirs["metrics"] / f"{base_name}_summary.csv"
    series_csv_path = dirs["metrics"] / f"{base_name}_vision_timeseries.csv"
    audio_csv_path = dirs["metrics"] / f"{base_name}_speaking_activity.csv"
    md_path = dirs["reports"] / f"{base_name}_report.md"
    html_path = dirs["reports"] / f"{base_name}_report.html"

    summary = result.get("summary", {})
    vision_series = result.get("vision", {}).get("series") or []
    speaking_activity = result.get("audio", {}).get("speaking_activity") or []

    write_json(json_path, result)
    _write_csv(summary_csv_path, [summary], fieldnames=list(summary.keys()) or ["metric"])
    _write_csv(
        series_csv_path,
        vision_series,
        fieldnames=["frame", "timestamp", "person_count", "movement_energy", "stationary_ratio", "transition_intensity"],
    )
    if speaking_activity:
        _write_csv(audio_csv_path, speaking_activity, fieldnames=["timestamp", "speaking_seconds"])

    markdown_report = generate_markdown_report(result)
    md_path.write_text(markdown_report, encoding="utf-8")
    html_path.write_text(generate_html_report(markdown_report, result), encoding="utf-8")

    exports = {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "vision_timeseries_csv": str(series_csv_path),
        "markdown_report": str(md_path),
        "html_report": str(html_path),
    }
    if speaking_activity:
        exports["speaking_activity_csv"] = str(audio_csv_path)
    return exports
