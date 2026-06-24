# Classroom Intelligence Demo

A lean Streamlit prototype that explores a Smart Education and Classroom Intelligence concept. The demo uses local classroom clips to produce anonymous aggregate analytics: occupancy, movement energy, transition intensity, stationary/seated proxies, speech cadence, and exportable reports.

This prototype uses pretrained YOLOv8n inference and rule-based demo proxy metrics. It has not been trained or fine-tuned on the four classroom clips; those clips are local demo inputs only.

## Current Demo Status

- Streamlit dashboard: working
- Media detection: 4 clips detected
- Vision analytics: YOLOv8n pretrained inference
- Tracking: ByteTrack where available through Ultralytics
- Training: not performed, not needed for the MVP
- GPU: active when CUDA-enabled PyTorch is installed in `.venv`
- Audio: extraction works, transcription requires Whisper install
- Privacy: no face recognition, no identity recognition, aggregate analytics only

## What It Demonstrates

- Camera inputs: local classroom video clips are discovered automatically.
- Computer vision analytics: YOLO person detection with ByteTrack when available.
- Speech analytics: audio extraction and optional Whisper transcription.
- Teacher analytics: speech cadence and movement/circulation proxies.
- Student analytics: aggregate occupancy and movement/stationary proxies only.
- Dashboard insights: KPI cards, charts, annotated videos, and an AI insight summary.
- Reports: JSON, CSV, Markdown, and HTML outputs per analyzed clip.

This is a demonstration prototype, not a research-grade classroom model. The metrics are explainable proxies designed to make the platform concept concrete.

## Project Structure

```text
classroom-analytics-demo/
  app.py
  requirements.txt
  README.md
  config/
    scenarios.yaml
  src/
    media_index.py
    vision_analyzer.py
    audio_analyzer.py
    metrics.py
    report_generator.py
    utils.py
  outputs/
    annotated_videos/
    reports/
    metrics/
  scripts/
    setup_gpu_windows.ps1
    setup_audio.ps1
    run_validation.py
```

The current folder can keep the existing videos at the root. The app also scans these folders recursively when present: `data`, `videos`, `media`, `assets`, `clips`, and `input`.

## Setup

Use Python 3.10 or newer. For the RTX 5090 demo machine, the recommended path is the GPU setup script:

```powershell
cd "C:\Users\AUT\Desktop\classroom analyics"
.\scripts\setup_gpu_windows.ps1
.\scripts\setup_audio.ps1
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The quick demo command sequence is:

```powershell
cd "C:\Users\AUT\Desktop\classroom analyics"
.\scripts\setup_gpu_windows.ps1
.\scripts\setup_audio.ps1
streamlit run app.py
```

The app prints runtime device information at startup. If CUDA is available through PyTorch, YOLO and Whisper-compatible paths will prefer GPU. If CUDA is not available, the app runs on CPU with slower processing.

## RTX 5090 / CUDA Note

This machine has an NVIDIA RTX 5090, but GPU use depends on the Python environment having a CUDA-enabled PyTorch build. Check it with:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The default GPU setup script uses:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If that fails, check the official PyTorch selector for the current CUDA wheel that matches the installed NVIDIA driver. The dashboard automatically switches to `cuda` once `torch.cuda.is_available()` is true.

At startup the app prints and displays:

- Python version
- torch version
- `torch.cuda.is_available()`
- CUDA version seen by torch
- GPU name
- Ultralytics version

## Media Mapping

The demo reads `config/scenarios.yaml` for scenario definitions and manual clip mapping. The existing local clips are mapped as:

- `Task1_Kids_Walking.webm`: Kids Walking / Transitions
- `Task2_4_CCTV_Chairs_Moving.mkv`: Kids Seated / Independent Learning and Teacher Moving
- `Task3_Teacher_Speaking.webm`: Teacher Speaking
- `Task5_Aggregate_Classroom.mkv`: Aggregate Classroom Dynamics

If a future filename is unclear, add it under `manual_mappings` in `config/scenarios.yaml`.

## Audio / Whisper

Audio extraction uses `ffmpeg`. The included `imageio-ffmpeg` package usually provides an ffmpeg binary even when system ffmpeg is not on PATH.

Transcription is optional. Install it with:

```powershell
.\scripts\setup_audio.ps1
```

The app prefers `openai-whisper` on CUDA when CUDA-enabled PyTorch is active, and keeps `faster-whisper` as a fallback. The optional audio packages are listed in `requirements-audio.txt`.

Manual install commands:

```powershell
python -m pip install openai-whisper
python -m pip install faster-whisper
```

When neither Whisper package is installed, the dashboard still runs and reports that audio extraction is available but transcription is missing.

## Outputs

Each run writes files under `outputs`:

- `outputs/annotated_videos`: processed MP4 videos with anonymous person boxes, IDs, and overlay metrics.
- `outputs/metrics`: JSON metrics, summary CSV, vision time series CSV, and media manifest.
- `outputs/reports`: Markdown and HTML report for the analyzed clip.

Run validation across all four clips:

```powershell
.\.venv\Scripts\python.exe scripts\run_validation.py --preset Medium --full
```

This generates full annotated videos, JSON metrics, CSV files, and reports for each classroom clip.

## Demo Mode

For a smooth screen recording, the sidebar includes a **Demo Mode** control that
loads a previously generated result instantly — repopulating the KPI cards, AI
insight, annotated video, charts, and export downloads without rerunning
YOLO/Whisper. Pre-run each clip once (or use the existing `outputs/`), then in
the sidebar pick a saved result and click `Load saved result`. A banner marks the
dashboard as showing a saved result; `Run Analysis` still performs a fresh
computation with the identical pipeline.

## Demo Script

A full recording-ready narration and shot list lives in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).
Quick version:

1. Launch with `streamlit run app.py`.
2. Select a scenario in the sidebar.
3. Pick the matching classroom clip.
4. Use `Medium` processing for a quick but credible run, or use `Demo Mode` to load a saved result instantly.
5. Click `Run Analysis` (or `Load saved result`).
6. Show the original clip, annotated output, KPI cards, charts, and AI Insight Summary.
7. Download the HTML report and JSON/CSV metrics to demonstrate exportability.

## Privacy Note

This prototype does not perform face recognition, student identification, identity matching, emotion recognition, or child profiling. All outputs are anonymous aggregate analytics intended for classroom-level operational insight.

## Pretrained Inference, Not Training

The four demo videos are too small and not labelled, so they are not used to train or fine-tune a classroom AI model. The MVP intentionally uses pretrained YOLOv8n person detection, Ultralytics tracking, Whisper speech transcription, and transparent proxy metrics for demonstration purposes.

## Demo-Only Proxy Metrics

- `movement_energy`: normalized tracked centroid displacement across sampled frames.
- `stationary_ratio`: proxy for seated/still behavior based on low centroid displacement.
- `transition_intensity`: combined movement and occupancy signal.
- `teacher_movement_proxy`: aggregate movement proxy when teacher-specific classification is not available.
- `classroom_state`: rule-based label inferred from occupancy, movement, stationary ratio, and speech pace.

These are suitable for a demonstration prototype and should be validated or replaced with domain-specific models before production use.
