from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.audio_analyzer import analyze_audio
from src.demo_loader import load_saved_results
from src.media_index import build_media_manifest, load_config, media_for_scenario, save_media_manifest, scenario_options
from src.metrics import summarize_analysis
from src.report_generator import save_all_exports
from src.utils import configure_local_runtime_dirs, ensure_output_dirs, print_runtime_banner, project_root, relative_to_root
from src.vision_analyzer import VisionAnalyzer


ROOT = project_root()
configure_local_runtime_dirs(ROOT)


st.set_page_config(
    page_title="Classroom Intelligence Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def runtime_info() -> dict[str, Any]:
    return print_runtime_banner()


@st.cache_data(ttl=20)
def load_manifest(root: str) -> list[dict[str, Any]]:
    manifest = build_media_manifest(Path(root), include_durations=True)
    save_media_manifest(manifest, Path(root))
    return manifest


@st.cache_data(ttl=20)
def load_demo_results(root: str) -> list[dict[str, Any]]:
    return load_saved_results(Path(root))


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --ink: #13222d;
            --muted: #61717d;
            --line: #d8e1e6;
            --panel: #ffffff;
            --wash: #eef5f3;
            --teal: #1f7a6d;
            --amber: #c98120;
            --blue: #24527a;
          }
          .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #edf5f2 0%, #f8fafb 100%);
            border-right: 1px solid var(--line);
          }
          .demo-header {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.1rem;
            padding-bottom: 0.9rem;
          }
          .demo-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.7rem);
            line-height: 0.95;
            font-weight: 820;
            letter-spacing: 0;
            margin: 0;
          }
          .demo-subtitle {
            color: var(--muted);
            max-width: 980px;
            font-size: 1.02rem;
            line-height: 1.45;
            margin-top: 0.65rem;
          }
          .kpi-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 15px 12px;
            min-height: 105px;
            box-shadow: 0 10px 24px rgba(24, 38, 48, 0.055);
          }
          .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 720;
            margin-bottom: 9px;
          }
          .kpi-value {
            color: var(--ink);
            font-size: 1.65rem;
            line-height: 1.05;
            font-weight: 790;
            overflow-wrap: anywhere;
          }
          .insight-box {
            background: #e7f3ef;
            border: 1px solid #bddbd3;
            border-left: 6px solid var(--teal);
            border-radius: 8px;
            padding: 18px 20px;
            color: var(--ink);
            font-size: 1.02rem;
            line-height: 1.55;
          }
          .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            background: #e8f0f5;
            color: #24455f;
            font-weight: 700;
            font-size: 0.82rem;
            margin-right: 7px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str | int | float) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def media_label(item: dict[str, Any]) -> str:
    duration = item.get("duration_seconds")
    duration_text = f" | {duration:.0f}s" if isinstance(duration, (int, float)) and duration else ""
    return f"{item['name']} | {item['size_mb']} MB{duration_text}"


def render_chart(data: pd.DataFrame, x: str, y: str, title: str, color: str = "#1f7a6d") -> None:
    if data.empty or x not in data or y not in data:
        st.info(f"No data available for {title.lower()}.")
        return
    try:
        import plotly.express as px

        fig = px.line(data, x=x, y=y, title=title)
        fig.update_traces(line_color=color, line_width=3)
        fig.update_layout(
            height=310,
            margin=dict(l=20, r=20, t=52, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.72)",
            font=dict(color="#13222d"),
        )
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.subheader(title)
        st.line_chart(data.set_index(x)[y])


def run_analysis(
    selected_media: dict[str, Any],
    scenario_id: str,
    scenario: dict[str, Any],
    frame_stride: int,
    max_seconds: int | None,
    model_name: str,
    confidence: float,
    run_audio: bool,
) -> dict[str, Any]:
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    def update_progress(progress: float, message: str) -> None:
        progress_bar.progress(min(max(progress, 0.0), 1.0))
        status_placeholder.info(message)

    media_path = Path(selected_media["path"])
    vision_result: dict[str, Any]
    if selected_media.get("kind") == "video":
        analyzer = VisionAnalyzer(model_name=model_name, confidence=confidence)
        vision_result = analyzer.analyze_video(
            media_path,
            scenario_id=scenario_id,
            frame_stride=frame_stride,
            max_seconds=max_seconds,
            output_root=ROOT,
            progress_callback=update_progress,
        )
    else:
        vision_result = {
            "status": "not_run",
            "message": "Static image analytics are not enabled in this lean demo.",
            "series": [],
            "warnings": ["Image preview is supported; analytics are focused on classroom video clips."],
        }

    status_placeholder.info("Checking audio track and transcription options")
    audio_result = analyze_audio(media_path, ROOT) if run_audio and selected_media.get("kind") == "video" else {
        "status": "not_run",
        "message": "Audio analytics skipped.",
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

    summary = summarize_analysis(vision_result, audio_result, scenario_id=scenario_id)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "media": selected_media,
        "scenario_id": scenario_id,
        "scenario": scenario,
        "vision": vision_result,
        "audio": audio_result,
        "summary": summary,
    }
    result["exports"] = save_all_exports(result, ROOT)
    progress_bar.progress(1.0)
    status_placeholder.success("Analysis complete")
    return result


def main() -> None:
    apply_theme()
    ensure_output_dirs(ROOT)
    info = runtime_info()
    config = load_config()
    scenarios = config.get("scenarios", {})
    scenario_labels = scenario_options(config)
    manifest = load_manifest(str(ROOT))

    st.markdown(
        """
        <div class="demo-header">
          <h1 class="demo-title">Classroom Intelligence Demo</h1>
          <div class="demo-subtitle">
            Anonymous camera and microphone analytics for classroom occupancy, movement, speech cadence,
            and exportable reporting.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Demo Controls")
        if info.get("cuda_available"):
            st.success(f"GPU: {info.get('gpu_name')}")
            st.caption(f"Vision model: YOLOv8n on CUDA {info.get('gpu_name')}")
        elif info.get("torch_available"):
            st.warning("GPU not active. Install CUDA-enabled PyTorch.")
            st.caption("Vision model: YOLOv8n on CPU")
        else:
            st.warning("GPU not active. Install CUDA-enabled PyTorch.")
            st.caption("Vision model: YOLOv8n requires PyTorch for GPU inference")

        st.caption(f"Python: {info.get('python_version')}")
        st.caption(f"torch: {info.get('torch_version') or 'not installed'}")
        st.caption(f"torch CUDA: {info.get('torch_cuda_version') or 'None'}")
        st.caption(f"torch.cuda.is_available(): {info.get('cuda_available')}")
        st.caption(f"ultralytics: {info.get('ultralytics_version') or 'not installed'}")
        st.info(
            "Pretrained YOLO inference only. The demo clips are not used for model training; "
            "analytics are anonymous proxy metrics."
        )

        if st.button("Refresh media index", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        scenario_id = st.selectbox(
            "Scenario",
            options=list(scenario_labels.keys()),
            format_func=lambda key: scenario_labels.get(key, key),
        )

        available_media = media_for_scenario(manifest, scenario_id)
        selected_media = None
        if available_media:
            selected_media = st.selectbox(
                "Media",
                options=available_media,
                format_func=media_label,
            )
        else:
            st.error("No local media files were found.")

        preset_name = st.selectbox("Processing preset", options=list(config.get("processing_presets", {}).keys()))
        preset = config.get("processing_presets", {}).get(preset_name, {"frame_stride": 5, "max_seconds": 90})
        frame_stride = int(st.number_input("Frame stride", min_value=1, max_value=30, value=int(preset["frame_stride"])))
        full_clip = st.checkbox("Analyze full clip", value=False)
        max_seconds = None if full_clip else int(
            st.number_input("Max seconds", min_value=10, max_value=900, value=int(preset["max_seconds"]), step=10)
        )
        model_name = st.text_input("YOLO model", value="yolov8n.pt")
        confidence = float(st.slider("Detection confidence", min_value=0.1, max_value=0.8, value=0.25, step=0.05))
        run_audio = st.checkbox("Run audio analytics", value=True)
        analyze_clicked = st.button("Run Analysis", type="primary", width="stretch", disabled=not selected_media)

        st.markdown("---")
        st.caption("Demo Mode — instant playback of a saved result, no recompute")
        demo_results = load_demo_results(str(ROOT))
        if demo_results:
            demo_index = st.selectbox(
                "Saved result",
                options=list(range(len(demo_results))),
                format_func=lambda i: demo_results[i]["demo_label"],
            )
            if st.button("Load saved result", width="stretch"):
                st.session_state["analysis_result"] = demo_results[demo_index]
                st.session_state["result_source"] = "demo"
                st.rerun()
        else:
            st.caption("No saved results yet. Run an analysis to populate Demo Mode.")

    if not manifest:
        st.warning("No videos or images were found in the project folder. Add media under data, videos, media, assets, clips, or input.")
        return

    if selected_media is None:
        return

    scenario = scenarios.get(scenario_id, {})
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Original Media")
        if selected_media["kind"] == "video":
            st.video(selected_media["path"])
        else:
            st.image(selected_media["path"], width="stretch")
    with right:
        st.subheader(scenario.get("title", "Scenario"))
        st.write(scenario.get("description", ""))
        st.markdown(
            " ".join(
                f'<span class="status-pill">{metric}</span>'
                for metric in scenario.get("expected_metrics", [])
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"Selected file: {relative_to_root(selected_media['path'], ROOT)}")

    if analyze_clicked:
        with st.status("Running classroom analytics", expanded=True):
            st.session_state["analysis_result"] = run_analysis(
                selected_media=selected_media,
                scenario_id=scenario_id,
                scenario=scenario,
                frame_stride=frame_stride,
                max_seconds=max_seconds,
                model_name=model_name,
                confidence=confidence,
                run_audio=run_audio,
            )
            st.session_state["result_source"] = "live"

    result = st.session_state.get("analysis_result")
    if not result:
        with st.expander("Media Manifest", expanded=False):
            st.dataframe(pd.DataFrame(manifest), width="stretch", hide_index=True)
        return

    summary = result.get("summary", {})
    st.divider()
    if st.session_state.get("result_source") == "demo":
        st.info(
            f"Demo Mode — showing a previously saved result: {result.get('demo_label', '')}. "
            "Use Run Analysis in the sidebar for a fresh computation."
        )
    st.subheader("KPI Summary")
    row_one = st.columns(4)
    with row_one[0]:
        kpi_card("Average Occupancy", summary.get("average_occupancy", 0))
    with row_one[1]:
        kpi_card("Peak Occupancy", summary.get("peak_occupancy", 0))
    with row_one[2]:
        kpi_card("Movement Energy", summary.get("movement_energy", 0))
    with row_one[3]:
        kpi_card("Classroom State", summary.get("classroom_state", "Low Activity"))

    row_two = st.columns(4)
    with row_two[0]:
        kpi_card("Stationary Proxy", f"{float(summary.get('stationary_ratio', 0)):.0%}")
    with row_two[1]:
        kpi_card("Transition Intensity", summary.get("transition_intensity", 0))
    with row_two[2]:
        kpi_card("Speaking Pace", f"{summary.get('speaking_pace_wpm', 0)} WPM")
    with row_two[3]:
        kpi_card("Word Count", summary.get("word_count", 0))

    st.subheader("AI Insight Summary")
    st.markdown(f'<div class="insight-box">{summary.get("ai_insight_summary", "")}</div>', unsafe_allow_html=True)

    audio = result.get("audio", {})
    if audio.get("status") == "ok" and audio.get("word_count"):
        st.subheader("Speech Insights")
        st.caption("From the speaker's audio only — aggregate signals, no speaker identification or student responses.")
        speech_cols = st.columns(4)
        with speech_cols[0]:
            kpi_card("Questions Posed", audio.get("question_count", 0))
        with speech_cols[1]:
            kpi_card("Talk Ratio", f"{float(audio.get('talk_ratio', 0)):.0%}")
        with speech_cols[2]:
            kpi_card("Unique Words", audio.get("unique_words", 0))
        with speech_cols[3]:
            kpi_card("Avg Sentence", f"{audio.get('avg_words_per_sentence', 0)} words")
        keywords = audio.get("keywords", []) or []
        if keywords:
            st.markdown(
                "**Lesson keywords:** "
                + " ".join(f'<span class="status-pill">{keyword}</span>' for keyword in keywords),
                unsafe_allow_html=True,
            )
        sample_questions = audio.get("sample_questions", []) or []
        if sample_questions:
            with st.expander("Sample questions posed by the speaker", expanded=False):
                for question in sample_questions:
                    st.markdown(f"- {question}")

    annotated_path = result.get("vision", {}).get("annotated_video_path")
    if annotated_path:
        st.subheader("Annotated Processed Video")
        st.video(annotated_path)

    vision_series = pd.DataFrame(result.get("vision", {}).get("series") or [])
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        render_chart(vision_series, "timestamp", "person_count", "Person Count Over Time", "#24527a")
    with chart_right:
        render_chart(vision_series, "timestamp", "movement_energy", "Movement Energy Over Time", "#1f7a6d")

    speaking_activity = pd.DataFrame(result.get("audio", {}).get("speaking_activity") or [])
    if not speaking_activity.empty:
        render_chart(speaking_activity, "timestamp", "speaking_seconds", "Speaking Activity Over Time", "#c98120")

    warnings = list(result.get("vision", {}).get("warnings") or []) + list(result.get("audio", {}).get("warnings") or [])
    if warnings:
        with st.expander("Runtime Notes", expanded=False):
            for warning in warnings:
                st.warning(warning)

    st.subheader("Exports")
    export_cols = st.columns(5)
    for index, (label, path_value) in enumerate(result.get("exports", {}).items()):
        path = Path(path_value)
        if not path.exists():
            continue
        with export_cols[index % len(export_cols)]:
            st.download_button(
                label=label.replace("_", " ").title(),
                data=path.read_bytes(),
                file_name=path.name,
                width="stretch",
            )

    with st.expander("Media Manifest", expanded=False):
        st.dataframe(pd.DataFrame(manifest), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
