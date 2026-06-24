"""Bundle client-ready deliverables into a single zip.

Collects the polished demo video plus the latest annotated (H.264) processed
clip and report for each scenario, gives them clean human-readable names, and
zips them so the whole set can be sent to a partner in one file.

Usage:
    .venv/Scripts/python.exe scripts/package_deliverables.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DELIV = OUT / "deliverables"

# Source media name -> clean deliverable name
CLEAN = {
    "Task1_Kids_Walking.webm": "1_Kids_Walking_Transitions",
    "Task2_4_CCTV_Chairs_Moving.mkv": "2_Seated_Independent_Learning",
    "Task3_Teacher_Speaking.webm": "3_Teacher_Speaking",
    "Task5_Aggregate_Classroom.mkv": "4_Aggregate_Classroom_Dynamics",
}


def latest_per_media() -> dict[str, tuple[float, dict, str]]:
    best: dict[str, tuple[float, dict, str]] = {}
    for f in glob.glob(str(OUT / "metrics" / "*_metrics.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        name, mt = d.get("media", {}).get("name"), os.path.getmtime(f)
        if name in CLEAN and (name not in best or mt > best[name][0]):
            best[name] = (mt, d, f)
    return best


def main() -> None:
    if DELIV.exists():
        shutil.rmtree(DELIV)
    (DELIV / "annotated_videos").mkdir(parents=True, exist_ok=True)
    (DELIV / "reports").mkdir(parents=True, exist_ok=True)

    best = latest_per_media()
    for name, clean in CLEAN.items():
        if name not in best:
            print("skip (no analysis yet):", name)
            continue
        _mt, d, mfile = best[name]
        av = d.get("vision", {}).get("annotated_video_path")
        if av and Path(av).exists():
            shutil.copy2(av, DELIV / "annotated_videos" / f"{clean}_annotated.mp4")
            print("  + annotated:", clean)
        base = Path(mfile).name[: -len("_metrics.json")]
        for ext in ("report.html", "report.md"):
            rp = OUT / "reports" / f"{base}_{ext}"
            if rp.exists():
                shutil.copy2(rp, DELIV / "reports" / f"{clean}_{ext}")

    demo = OUT / "demo_video" / "Classroom_Intelligence_Demo.mp4"
    if demo.exists():
        shutil.copy2(demo, DELIV / "Classroom_Intelligence_Demo.mp4")
        print("  + demo video")

    archive = shutil.make_archive(str(OUT / "Classroom_Intelligence_client_package"), "zip", DELIV)
    print(f"\nPACKAGE -> {archive}  ({os.path.getsize(archive) / 1e6:.1f} MB)")
    print("Contents:")
    for p in sorted(DELIV.rglob("*")):
        if p.is_file():
            print("  -", p.relative_to(DELIV).as_posix(), f"({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
