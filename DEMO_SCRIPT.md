# Classroom Intelligence Demo — Recording Script

A recording-ready narration and shot list for the client/RFP demo video.
Target length: **2:30–3:30**. Tone: confident, concrete, no overclaiming.

## Before You Record

- Launch the app from the GPU venv so the sidebar shows the RTX 5090 badge:
  ```powershell
  cd "C:\Users\AUT\Desktop\classroom analyics"
  .\.venv\Scripts\Activate.ps1
  streamlit run app.py
  ```
- Confirm the sidebar shows **GPU: NVIDIA GeForce RTX 5090** (green) and
  `torch.cuda.is_available(): True`.
- Pre-run all four clips once (or use the already-generated outputs) so
  **Demo Mode** can load results instantly — no waiting on camera during the take.
- Set browser zoom so the title, KPI row, and a chart fit cleanly on screen.
- Close other tabs/notifications. Record at 1080p, full-window.

## Shot List & Narration

### 1. Open — title and positioning (0:00–0:20)
**On screen:** Dashboard landing page. Title *Classroom Intelligence Demo* and
the subtitle visible.
**Say:**
> "This is our Classroom Intelligence demo — a working prototype that turns
> ordinary classroom camera and microphone footage into anonymous, aggregate
> analytics. Everything you'll see runs on local clips, with no student
> identification of any kind."

### 2. Technical credibility — sidebar (0:20–0:40)
**On screen:** Hover/point to the sidebar: green GPU badge, torch/CUDA lines,
the "Pretrained YOLO inference only" note.
**Say:**
> "It runs on an NVIDIA RTX 5090 using pretrained YOLOv8 person detection and
> ByteTrack tracking, with optional Whisper speech analytics. We're not training
> on these clips — this is pretrained inference, and every metric is an
> explainable proxy, not a black box."

### 3. Scenario + original clip (0:40–1:05)
**On screen:** In the sidebar, select **Teacher Speaking**. The original clip
plays in the left panel; the scenario card and metric pills show on the right.
**Say:**
> "We organize the demo around classroom scenarios — kids walking and
> transitioning, seated independent work, the teacher speaking, and aggregate
> classroom dynamics. Here's the teacher-speaking scenario with the original
> clip on the left and the metrics we expect to surface on the right."

### 4. Load results instantly — Demo Mode (1:05–1:20)
**On screen:** Scroll the sidebar to **Demo Mode**, pick the matching saved
result, click **Load saved result**. The dashboard fills in immediately.
**Say:**
> "For a live demo we use Demo Mode, which loads a previously processed result
> instantly — so we can review the analytics without waiting on a fresh
> computation. A full run uses the exact same pipeline."

*(Alternative for a live processing take: choose a Medium preset and click
**Run Analysis** instead, then narrate the progress bar.)*

### 5. KPI cards (1:20–1:50)
**On screen:** The KPI Summary grid.
**Say:**
> "At the top we get classroom-level KPIs: average and peak occupancy, a
> movement-energy proxy, a stationary or seated proxy, transition intensity,
> and — for the teacher clip — speaking pace and word count. The system reads
> this as a *Teacher-led* classroom state, driven by the speech pacing with
> movement staying secondary to instruction."

### 6. AI insight + annotated video (1:50–2:20)
**On screen:** The AI Insight Summary box, then scroll to the annotated video
and play a few seconds showing the anonymous person boxes and tracking IDs.
**Say:**
> "The AI Insight Summary turns those numbers into a plain-language read for a
> non-technical stakeholder. And here's the annotated video — anonymous
> bounding boxes and tracking IDs only. No faces, no names, no identity
> matching."

### 7. Charts (2:20–2:40)
**On screen:** Person Count Over Time and Movement Energy Over Time; for the
teacher clip, the Speaking Activity chart.
**Say:**
> "The time-series charts show how occupancy and movement evolve across the
> clip, and for audio scenarios, when speech is most active — useful signals
> for classroom flow, supervision, and engagement at the room level."

### 8. Exports (2:40–2:55)
**On screen:** The Exports row; hover the JSON, CSV, Markdown, and HTML report
buttons. Optionally open the HTML report in a new tab.
**Say:**
> "Every run is exportable — JSON and CSV for your data team, plus a clean
> Markdown and HTML report for stakeholders. Nothing is locked inside the tool."

### 9. Close — privacy (2:55–3:15)
**On screen:** Return to the top, or show the report's privacy note / the
sidebar pretrained-only note.
**Say:**
> "To be clear on privacy: this prototype does no face recognition, no student
> identification, and no emotion detection. It produces aggregate,
> privacy-preserving analytics for classroom-level operational insight — and
> it's a starting point we'd validate and tailor to your environment."

## Notes for the Presenter

- These metrics are **demo proxies**, not validated classroom science. If asked
  about accuracy, say they're explainable proxies designed to make the platform
  concept concrete, and would be validated/replaced with domain models for
  production.
- Keep claims at the level of "signals" and "proxies," not "measurements."
- If a clip looks sparse (low occupancy), that's expected for short windows —
  lean on the teacher-speaking and aggregate clips for the richest screen.
