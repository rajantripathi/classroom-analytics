# Claude Code Handoff: Classroom Intelligence Demo

## North Star

Build a credible, client-facing Classroom Intelligence Demo for an RFP/proposal conversation. The goal is not research-grade AI perfection; the goal is a polished, explainable, working prototype that makes the Smart Education / Classroom Analytics platform feel real in a client demo video.

The demo should look professional, stable, and believable. Prioritize clarity, clean UI, smooth demo flow, and strong narrative over adding complex new AI features.

## Client-Facing Positioning

This is a pretrained inference demo:

- Vision uses pretrained YOLOv8n person detection through Ultralytics.
- Tracking uses ByteTrack where available.
- Audio uses Whisper transcription when installed.
- Metrics are anonymous demo proxies, not validated classroom science.
- No model training or fine-tuning is performed on the four demo clips.
- No face recognition, identity recognition, child identification, or emotion recognition.
- All analytics are aggregate and privacy-preserving.

Keep this language visible in the README and, where useful, in the UI.

## Current Project State

Project folder:

```powershell
C:\Users\AUT\Desktop\classroom analyics
```

Main app:

```powershell
streamlit run app.py
```

Recommended run command:

```powershell
cd "C:\Users\AUT\Desktop\classroom analyics"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The `.venv` has been created and CUDA PyTorch was installed successfully.

Verified GPU status:

```text
torch: 2.11.0+cu128
torch CUDA: 12.8
torch.cuda.is_available(): True
GPU: NVIDIA GeForce RTX 5090
```

## Key Files

- `app.py`: Streamlit dashboard.
- `src/vision_analyzer.py`: YOLOv8n inference, ByteTrack, annotated video output, movement metrics.
- `src/audio_analyzer.py`: audio extraction and Whisper transcription.
- `src/media_index.py`: local media discovery and scenario mapping.
- `src/metrics.py`: proxy KPI and classroom state logic.
- `src/report_generator.py`: JSON, CSV, Markdown, and HTML exports.
- `config/scenarios.yaml`: scenario definitions and manual clip mappings.
- `scripts/setup_gpu_windows.ps1`: CUDA PyTorch setup for RTX 5090.
- `scripts/setup_audio.ps1`: optional Whisper setup.
- `scripts/run_validation.py`: runs validation across all local clips.
- `README.md`: setup, demo status, privacy, and no-training notes.

## Existing Demo Clips

The app detects four clips:

- `Task1_Kids_Walking.webm`: Kids Walking / Transitions.
- `Task2_4_CCTV_Chairs_Moving.mkv`: Kids Seated / Independent Learning and Teacher Moving.
- `Task3_Teacher_Speaking.webm`: Teacher Speaking.
- `Task5_Aggregate_Classroom.mkv`: Aggregate Classroom Dynamics.

Full validation has already generated annotated outputs under:

```text
outputs/annotated_videos
outputs/metrics
outputs/reports
```

Latest full annotated videos include:

- `Task5_Aggregate_Classroom_20260624_150950_annotated.mp4`
- `Task2_4_CCTV_Chairs_Moving_20260624_151048_annotated.mp4`
- `Task1_Kids_Walking_20260624_151103_annotated.mp4`
- `Task3_Teacher_Speaking_20260624_151229_annotated.mp4`

The teacher-speaking clip produced a Whisper transcript successfully.

## Demo Video Goal

We will create a short client demo video that shows the platform concept clearly:

1. Open with the Streamlit dashboard title: Classroom Intelligence Demo.
2. Show GPU/runtime status in the sidebar to reinforce technical credibility.
3. Select each scenario and show the matching original classroom clip.
4. Run or show completed analysis.
5. Show annotated video with anonymous boxes and tracking IDs.
6. Highlight KPI cards: average occupancy, peak occupancy, movement energy, stationary proxy, transition intensity, speaking pace, word count, classroom state.
7. Show charts and the AI Insight Summary.
8. Show export/report capability: JSON, CSV, Markdown, HTML.
9. End with the privacy message: aggregate analytics only, no identity recognition.

The video should feel like a credible RFP prototype, not a hackathon script. Avoid overclaiming accuracy.

## Suggested Next Steps For Claude

1. Run the app from `.venv` and visually inspect the UI.
2. Polish any rough Streamlit layout issues that appear in the browser.
3. Make the demo story smoother for screen recording.
4. Add a “Demo Mode” option if useful, showing latest generated outputs without rerunning long analyses.
5. Review output reports for client-friendly language.
6. Prepare a short narration/script for the demo video.
7. Do not add training, face recognition, student identification, or emotion detection.

## GitHub Recommendation

Yes, it is a good idea to push the code to GitHub, preferably to a private repository, but do not push the large video clips, generated annotated videos, model weights, or client-sensitive outputs directly.

Recommended:

- Push source code, config, scripts, README, and this handoff file.
- Keep `outputs/`, `.venv/`, `yolov8n.pt`, and raw demo videos out of normal Git.
- If video assets must be versioned, use Git LFS or a private shared drive.
- Keep the repository private unless all media rights and client-sensitivity questions are cleared.
- Add a small sample manifest or placeholder instructions instead of committing large classroom media.

The existing `.gitignore` already excludes many generated files, but check it before the first commit.
