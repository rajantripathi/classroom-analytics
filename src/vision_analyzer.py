from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from .utils import configure_local_runtime_dirs, ensure_output_dirs, get_device_info, safe_slug, timestamp_slug


ProgressCallback = Callable[[float, str], None]


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0) -> None:
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            self.tracks = {}
            return detections

        assigned_tracks: set[int] = set()
        updated_tracks: dict[int, tuple[float, float]] = {}
        enriched: list[dict[str, Any]] = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            best_id = None
            best_distance = float("inf")

            for track_id, previous_center in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                distance = math.dist(center, previous_center)
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None or best_distance > self.max_distance:
                best_id = self.next_id
                self.next_id += 1

            assigned_tracks.add(best_id)
            updated_tracks[best_id] = center
            enriched_detection = {**detection, "track_id": best_id}
            enriched.append(enriched_detection)

        self.tracks = updated_tracks
        return enriched


class VisionAnalyzer:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.device_info = get_device_info()
        self.device = "cuda" if self.device_info.get("cuda_available") else "cpu"
        self.device_arg: int | str = 0 if self.device == "cuda" else "cpu"
        self.model: Any | None = None
        self.model_error: str | None = None
        self.detector_name = "not_loaded"
        self.centroid_tracker = CentroidTracker()
        self.previous_centers: dict[int, tuple[float, float]] = {}

    def _load_yolo(self) -> Any | None:
        if self.model is not None:
            return self.model
        if self.model_error:
            return None
        try:
            configure_local_runtime_dirs()
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
            device_label = self.device_info.get("gpu_name") if self.device == "cuda" else "CPU"
            self.detector_name = f"YOLOv8n pretrained ({self.model_name}) on {device_label}"
            print(f"[Classroom Intelligence Demo] YOLO device: {self.device_arg} ({device_label})")
            return self.model
        except Exception as exc:
            self.model_error = str(exc)
            self.detector_name = "opencv_hog_fallback"
            return None

    def _detect_with_yolo(self, frame: Any) -> list[dict[str, Any]] | None:
        model = self._load_yolo()
        if model is None:
            return None

        try:
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=self.confidence,
                device=self.device_arg,
                verbose=False,
            )
        except Exception:
            results = model.predict(
                frame,
                classes=[0],
                conf=self.confidence,
                device=self.device_arg,
                verbose=False,
            )

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)
        raw_ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

        detections: list[dict[str, Any]] = []
        for bbox, confidence, track_id in zip(xyxy, confidences, raw_ids):
            x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                    "track_id": int(track_id) if track_id is not None else None,
                    "label": "person",
                }
            )

        if any(item["track_id"] is None for item in detections):
            detections = self.centroid_tracker.update([{**item, "track_id": None} for item in detections])
        return detections

    def _detect_with_hog(self, frame: Any) -> list[dict[str, Any]]:
        import cv2

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        detections = []
        for (x, y, width, height), confidence in zip(boxes, weights):
            detections.append(
                {
                    "bbox": [int(x), int(y), int(x + width), int(y + height)],
                    "confidence": float(confidence),
                    "track_id": None,
                    "label": "person",
                }
            )
        return self.centroid_tracker.update(detections)

    def _detect_persons(self, frame: Any) -> list[dict[str, Any]]:
        detections = self._detect_with_yolo(frame)
        if detections is not None:
            return detections
        return self._detect_with_hog(frame)

    def _movement_metrics(
        self,
        detections: list[dict[str, Any]],
        diagonal: float,
    ) -> tuple[float, float]:
        centers: dict[int, tuple[float, float]] = {}
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            centers[int(track_id)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        displacements: list[float] = []
        stationary_flags: list[bool] = []
        for track_id, center in centers.items():
            previous = self.previous_centers.get(track_id)
            if previous:
                normalised = math.dist(center, previous) / max(diagonal, 1.0)
                displacements.append(normalised)
                stationary_flags.append(normalised < 0.012)

        self.previous_centers = centers

        if not detections:
            return 0.0, 0.0
        if not displacements:
            return 0.0, 1.0

        movement_energy = min(1.0, sum(displacements))
        stationary_ratio = sum(1 for item in stationary_flags if item) / len(stationary_flags)
        return round(movement_energy, 4), round(stationary_ratio, 4)

    @staticmethod
    def _draw_annotations(frame: Any, detections: list[dict[str, Any]], overlay: dict[str, Any]) -> Any:
        import cv2

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection.get("track_id")
            confidence = detection.get("confidence", 0)
            color = (44, 160, 44) if track_id is None else (
                int(80 + (track_id * 47) % 150),
                int(170 + (track_id * 29) % 70),
                int(80 + (track_id * 83) % 150),
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}" if track_id is not None else "person"
            label = f"{label} {confidence:.2f}"
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 118, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        panel_height = 74
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], panel_height), (16, 24, 32), -1)
        lines = [
            f"People: {overlay.get('person_count', 0)}   Movement: {overlay.get('movement_energy', 0):.3f}   "
            f"Stationary: {overlay.get('stationary_ratio', 0):.0%}",
            f"State proxy: {overlay.get('state_hint', 'Analyzing')}   Anonymous aggregate analytics",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (16, 28 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (245, 248, 250),
                2,
                cv2.LINE_AA,
            )
        return annotated

    @staticmethod
    def _open_writer(output_path: Path, fps: float, width: int, height: int) -> Any | None:
        import cv2

        for codec in ("mp4v", "avc1", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
            writer.release()
        return None

    @staticmethod
    def _transcode_to_h264(path: Path, audio_source: Path | None = None) -> bool:
        """Re-encode the OpenCV output to browser-playable H.264, in place.

        OpenCV typically writes ``mp4v`` (MPEG-4 Part 2), which HTML5 video — and
        therefore Streamlit's ``st.video`` — cannot decode. We transcode to H.264
        (yuv420p, +faststart) with the bundled ffmpeg so the annotated clip plays
        in the dashboard and the demo video.

        When ``audio_source`` (the original clip) is given, its audio track is
        muxed in so the annotated output keeps the original sound. ``-shortest``
        trims audio to the (possibly capped) annotated video length; the optional
        audio map (``1:a:0?``) is a no-op for clips that have no audio.
        """
        try:
            import subprocess

            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = path.with_name(path.stem + "_h264.mp4")
            cmd = [ffmpeg, "-y", "-i", str(path)]
            if audio_source is not None:
                cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            cmd += ["-c:a", "aac", "-b:a", "128k"] if audio_source is not None else ["-an"]
            cmd += [str(tmp)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                path.unlink(missing_ok=True)
                tmp.replace(path)
                return True
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    def analyze_video(
        self,
        video_path: str | Path,
        scenario_id: str | None = None,
        frame_stride: int = 5,
        max_seconds: int | None = 90,
        output_root: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        try:
            import cv2
        except ImportError:
            return {
                "status": "error",
                "message": "opencv-python is not installed. Run pip install -r requirements.txt.",
                "series": [],
                "warnings": ["opencv-python is required for video analytics."],
            }

        self.previous_centers = {}
        self.centroid_tracker = CentroidTracker()
        path = Path(video_path)
        dirs = ensure_output_dirs(output_root)
        output_path = dirs["annotated_videos"] / f"{safe_slug(path.stem)}_{timestamp_slug()}_annotated.mp4"

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return {
                "status": "error",
                "message": f"Could not open video: {path}",
                "series": [],
                "warnings": ["OpenCV could not open the selected video."],
            }

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = total_frames / fps if total_frames and fps else 0.0
        max_frames = total_frames
        if max_seconds and max_seconds > 0:
            max_frames = min(total_frames, int(max_seconds * fps)) if total_frames else int(max_seconds * fps)

        writer = self._open_writer(output_path, fps, width, height) if width and height else None
        warnings: list[str] = []
        if writer is None:
            warnings.append("Annotated video writer could not be opened; metrics will still be computed.")
        if self.model_error:
            warnings.append(f"YOLO unavailable, using OpenCV HOG fallback: {self.model_error}")

        series: list[dict[str, Any]] = []
        last_detections: list[dict[str, Any]] = []
        last_overlay = {
            "person_count": 0,
            "movement_energy": 0.0,
            "stationary_ratio": 0.0,
            "state_hint": "Analyzing",
        }
        diagonal = math.hypot(width, height) or 1.0

        frame_index = 0
        processed_frames = 0
        while True:
            if max_frames and frame_index >= max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            should_process = frame_index % max(1, frame_stride) == 0
            if should_process:
                detections = self._detect_persons(frame)
                movement_energy, stationary_ratio = self._movement_metrics(detections, diagonal)
                person_count = len(detections)
                transition_intensity = min(100.0, movement_energy * 220.0 + person_count * 1.5)
                state_hint = "Transition" if transition_intensity >= 40 else "Independent Work"
                if person_count == 0:
                    state_hint = "Low Activity"

                sample = {
                    "frame": frame_index,
                    "timestamp": round(frame_index / fps, 2),
                    "person_count": person_count,
                    "movement_energy": movement_energy,
                    "stationary_ratio": stationary_ratio,
                    "transition_intensity": round(transition_intensity, 2),
                }
                series.append(sample)
                last_detections = detections
                last_overlay = {**sample, "state_hint": state_hint}
                processed_frames += 1

                if progress_callback and processed_frames % 3 == 0:
                    denominator = max(max_frames or total_frames or 1, 1)
                    progress_callback(min(frame_index / denominator, 0.99), f"Processed {processed_frames} sampled frames")

            if writer is not None:
                writer.write(self._draw_annotations(frame, last_detections, last_overlay))
            frame_index += 1

        capture.release()
        if writer is not None:
            writer.release()
            if output_path.exists() and output_path.stat().st_size > 0 and not self._transcode_to_h264(output_path, audio_source=path):
                warnings.append("Annotated video could not be re-encoded to H.264; it may not play in the browser.")

        if progress_callback:
            progress_callback(1.0, "Vision analysis complete")

        annotated_path = str(output_path) if output_path.exists() and output_path.stat().st_size > 0 else None
        if self.model_error and not any("YOLO unavailable" in item for item in warnings):
            warnings.append(f"YOLO unavailable, using OpenCV HOG fallback: {self.model_error}")

        return {
            "status": "ok",
            "message": "Vision analysis complete.",
            "input_path": str(path.resolve()),
            "annotated_video_path": annotated_path,
            "processed_frames": processed_frames,
            "analyzed_frames": frame_index,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration_seconds, 2),
            "analysis_duration_seconds": round(frame_index / fps, 2) if fps else 0,
            "frame_stride": frame_stride,
            "scenario_id": scenario_id,
            "detector": self.detector_name,
            "device": self.device,
            "yolo_device": self.device_arg,
            "device_info": self.device_info,
            "series": series,
            "warnings": warnings,
        }
