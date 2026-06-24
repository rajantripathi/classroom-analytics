from __future__ import annotations

import re
import subprocess
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import ensure_output_dirs, find_ffmpeg_executable, get_device_info, project_root, safe_slug


def _empty_audio_result(status: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "audio_path": None,
        "transcript": "",
        "segments": [],
        "speaking_activity": [],
        "word_count": 0,
        "words_per_minute": 0.0,
        "speaking_time_seconds": 0.0,
        "pause_count": 0,
        "pause_ratio": 0.0,
        "question_count": 0,
        "talk_ratio": 0.0,
        "unique_words": 0,
        "vocabulary_diversity": 0.0,
        "sentence_count": 0,
        "avg_words_per_sentence": 0.0,
        "keywords": [],
        "sample_questions": [],
        "lesson_summary": "",
        "warnings": [message] if message else [],
    }


def _extract_audio(video_path: Path, output_dir: Path) -> tuple[Path | None, str | None]:
    ffmpeg = find_ffmpeg_executable()
    if not ffmpeg:
        return None, (
            "ffmpeg was not found. Install ffmpeg or keep imageio-ffmpeg installed so audio "
            "can be extracted locally."
        )

    output_path = output_dir / f"{safe_slug(video_path.stem)}_audio.wav"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        if "does not contain any stream" in error.lower() or "audio" in error.lower():
            return None, "No usable audio track was detected in this video."
        return None, f"Audio extraction failed: {error[-500:]}"
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None, "Audio extraction produced an empty file."
    return output_path, None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


_STOPWORDS = set(
    (
        "the a an and or but if then so to of in on at for with as is are was were be been being it its "
        "this that these those i you he she we they them his her their our your my me him us do does did "
        "have has had not no yes will would can could should may might must just about into over under up "
        "down out off there here what when where why how which who whom while because now okay ok yeah "
        "right like get got going go also let lets us am pm one two three"
    ).split()
)


def _speech_insights(transcript: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Lightweight, explainable analysis of *what was said* (not who said it).

    Surfaces interactivity (questions), delivery (talk ratio), language richness,
    and the lesson's most prominent keywords — defensible, aggregate signals with
    no speaker identification.
    """
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    questions = [
        s
        for s in sentences
        if s.endswith("?")
        or re.match(
            r"^(what|why|how|when|where|who|which|can|could|do|does|did|is|are|will|would|should)\b",
            s.lower(),
        )
    ]
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    unique = set(words)
    duration = max((float(s.get("end", 0)) for s in segments), default=0.0)
    speaking = sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in segments)
    return {
        "question_count": len(questions),
        "talk_ratio": round(speaking / duration, 2) if duration else 0.0,
        "unique_words": len(unique),
        "vocabulary_diversity": round(len(unique) / len(words), 2) if words else 0.0,
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(len(words) / len(sentences), 1) if sentences else 0.0,
        "keywords": [w for w, _ in Counter(content).most_common(8)],
        "sample_questions": questions[:3],
    }


def _summarize_transcript(transcript: str, max_sentences: int = 2) -> str:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", transcript) if item.strip()]
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    ranked = sorted(sentences, key=lambda item: len(item.split()), reverse=True)
    selected = [sentence for sentence in sentences if sentence in set(ranked[:max_sentences])]
    return " ".join(selected[:max_sentences])


def _speaking_activity(segments: list[dict[str, Any]], bin_seconds: int = 5) -> list[dict[str, Any]]:
    if not segments:
        return []
    end_time = max(float(item.get("end", 0)) for item in segments)
    bins = int(end_time // bin_seconds) + 1
    activity = [{"timestamp": index * bin_seconds, "speaking_seconds": 0.0} for index in range(bins)]

    for segment in segments:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        cursor = start
        while cursor < end:
            bin_index = int(cursor // bin_seconds)
            bin_end = min(end, (bin_index + 1) * bin_seconds)
            if 0 <= bin_index < len(activity):
                activity[bin_index]["speaking_seconds"] += max(0.0, bin_end - cursor)
            cursor = bin_end

    for item in activity:
        item["speaking_seconds"] = round(item["speaking_seconds"], 2)
    return activity


def _transcribe_with_faster_whisper(audio_path: Path, device: str) -> tuple[list[dict[str, Any]], str]:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel("base", device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(str(audio_path), vad_filter=True)
    segments = [
        {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
        for segment in segments_iter
    ]
    transcript = " ".join(item["text"] for item in segments).strip()
    return segments, transcript


def _transcribe_with_openai_whisper(audio_path: Path, device: str) -> tuple[list[dict[str, Any]], str]:
    import numpy as np
    import whisper

    cache_dir = project_root() / "outputs" / "cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model("base", device=device, download_root=str(cache_dir))
    audio = _load_wav_mono_float32(audio_path)
    result = model.transcribe(audio.astype(np.float32), fp16=device == "cuda")
    segments = [
        {
            "start": float(segment.get("start", 0)),
            "end": float(segment.get("end", 0)),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in result.get("segments", [])
    ]
    transcript = str(result.get("text", "")).strip() or " ".join(item["text"] for item in segments).strip()
    return segments, transcript


def _load_wav_mono_float32(audio_path: Path) -> Any:
    import numpy as np

    with wave.open(str(audio_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def _transcribe(audio_path: Path, device: str) -> tuple[list[dict[str, Any]], str, str | None]:
    try:
        return (*_transcribe_with_openai_whisper(audio_path, device), None)
    except ImportError:
        pass
    except Exception as exc:
        return [], "", f"openai-whisper failed: {exc}"

    try:
        return (*_transcribe_with_faster_whisper(audio_path, device), None)
    except ImportError:
        return [], "", (
            "Whisper is not installed. Optional installs: "
            "pip install openai-whisper or pip install faster-whisper."
        )
    except Exception as exc:
        return [], "", f"faster-whisper failed: {exc}"


def _pause_metrics(segments: list[dict[str, Any]]) -> tuple[int, float]:
    if len(segments) < 2:
        return 0, 0.0

    pauses = []
    for previous, current in zip(segments, segments[1:]):
        gap = float(current.get("start", 0)) - float(previous.get("end", 0))
        if gap >= 0.7:
            pauses.append(gap)

    total_window = max(float(segments[-1].get("end", 0)) - float(segments[0].get("start", 0)), 0.01)
    pause_ratio = sum(pauses) / total_window
    return len(pauses), round(pause_ratio, 3)


def analyze_audio(video_path: str | Path, output_root: Path | None = None) -> dict[str, Any]:
    path = Path(video_path)
    dirs = ensure_output_dirs(output_root)
    audio_path, extraction_error = _extract_audio(path, dirs["audio"])
    if extraction_error:
        return _empty_audio_result("missing_audio_tool_or_track", extraction_error)
    if not audio_path:
        return _empty_audio_result("no_audio", "No audio was available for this clip.")

    device_info = get_device_info()
    device = "cuda" if device_info.get("cuda_available") else "cpu"
    segments, transcript, transcription_error = _transcribe(audio_path, device)
    if transcription_error:
        result = _empty_audio_result("audio_extracted_no_transcription", transcription_error)
        result["audio_path"] = str(audio_path)
        return result

    word_count = _word_count(transcript)
    speaking_time = sum(max(0.0, float(item.get("end", 0)) - float(item.get("start", 0))) for item in segments)
    words_per_minute = (word_count / (speaking_time / 60.0)) if speaking_time > 0 else 0.0
    pause_count, pause_ratio = _pause_metrics(segments)
    insights = _speech_insights(transcript, segments)

    return {
        "status": "ok",
        "message": "Audio transcription complete.",
        "audio_path": str(audio_path),
        "transcript": transcript,
        "segments": segments,
        "speaking_activity": _speaking_activity(segments),
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "speaking_time_seconds": round(speaking_time, 1),
        "pause_count": pause_count,
        "pause_ratio": pause_ratio,
        "lesson_summary": _summarize_transcript(transcript),
        **insights,
        "warnings": [],
    }
