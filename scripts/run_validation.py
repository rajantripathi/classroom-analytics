from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio_analyzer import analyze_audio
from src.media_index import build_media_manifest, load_config
from src.metrics import summarize_analysis
from src.report_generator import save_all_exports
from src.utils import print_runtime_banner, project_root
from src.vision_analyzer import VisionAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classroom demo validation across local clips.")
    parser.add_argument("--preset", default="Medium", help="Processing preset from config/scenarios.yaml")
    parser.add_argument("--full", action="store_true", help="Analyze full clips instead of preset max_seconds")
    parser.add_argument("--include-audio", action="store_true", help="Run audio analytics for every video")
    args = parser.parse_args()

    root = project_root()
    config = load_config(root / "config" / "scenarios.yaml")
    preset = config.get("processing_presets", {}).get(args.preset, {"frame_stride": 5, "max_seconds": 90})
    frame_stride = int(preset.get("frame_stride", 5))
    max_seconds = None if args.full else int(preset.get("max_seconds", 90))
    scenarios = config.get("scenarios", {})
    manifest = [item for item in build_media_manifest(root, include_durations=True) if item.get("kind") == "video"]

    print_runtime_banner()
    print(f"Found {len(manifest)} video clips")
    print(f"Preset: {args.preset} | frame_stride={frame_stride} | max_seconds={max_seconds or 'full clip'}")

    for index, media in enumerate(manifest, start=1):
        scenario_id = (media.get("scenario_ids") or ["aggregate_dynamics"])[0]
        scenario = scenarios.get(scenario_id, {"title": scenario_id})
        print(f"[{index}/{len(manifest)}] Analyzing {media['name']} as {scenario.get('title', scenario_id)}")

        analyzer = VisionAnalyzer(model_name="yolov8n.pt", confidence=0.25)
        vision = analyzer.analyze_video(
            media["path"],
            scenario_id=scenario_id,
            frame_stride=frame_stride,
            max_seconds=max_seconds,
            output_root=root,
        )

        should_run_audio = args.include_audio or scenario_id == "teacher_speaking"
        audio = analyze_audio(media["path"], root) if should_run_audio else {
            "status": "not_run",
            "message": "Audio analytics skipped for this validation clip.",
            "warnings": [],
            "word_count": 0,
            "words_per_minute": 0,
            "speaking_time_seconds": 0,
            "pause_count": 0,
            "pause_ratio": 0,
            "speaking_activity": [],
            "transcript": "",
            "lesson_summary": "",
        }

        summary = summarize_analysis(vision, audio, scenario_id=scenario_id)
        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(root),
            "media": media,
            "scenario_id": scenario_id,
            "scenario": scenario,
            "vision": vision,
            "audio": audio,
            "summary": summary,
        }
        exports = save_all_exports(result, root)
        print(f"  annotated: {vision.get('annotated_video_path')}")
        print(f"  state: {summary.get('classroom_state')} | exports: {', '.join(sorted(exports))}")
        if audio.get("status") != "ok" and should_run_audio:
            print(f"  audio: {audio.get('status')} - {audio.get('message')}")


if __name__ == "__main__":
    main()
