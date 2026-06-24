from __future__ import annotations

from statistics import mean
from typing import Any


def _round(value: float | int | None, digits: int = 2) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _average(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def classify_classroom_state(
    average_occupancy: float,
    movement_energy: float,
    stationary_ratio: float,
    words_per_minute: float,
    scenario_id: str | None = None,
) -> str:
    if average_occupancy <= 0.1 and words_per_minute <= 5:
        return "Low Activity"
    if scenario_id == "teacher_speaking" or words_per_minute >= 65:
        return "Teacher-led"
    if movement_energy >= 0.18:
        return "Transition"
    if average_occupancy >= 3 and movement_energy >= 0.07:
        return "Group Work"
    if stationary_ratio >= 0.62:
        return "Independent Work"
    return "Low Activity"


def build_ai_insight_summary(summary: dict[str, Any]) -> str:
    state = summary.get("classroom_state", "Low Activity")
    occupancy = summary.get("average_occupancy", 0)
    movement = summary.get("movement_energy", 0)
    stationary = summary.get("stationary_ratio", 0)
    wpm = summary.get("speaking_pace_wpm", 0)
    questions = summary.get("question_count", 0)
    keywords = summary.get("keywords", []) or []
    topic = ", ".join(keywords[:4])

    speech_extra = ""
    if questions:
        speech_extra += f" The teacher asked roughly {questions} questions, a marker of interactive instruction."
    if topic:
        speech_extra += f" Prominent lesson keywords include {topic}."

    if state == "Teacher-led":
        return (
            f"The clip shows a teacher-led pattern with about {occupancy:.1f} detected people on "
            f"average and speech pacing near {wpm:.0f} words per minute. Movement remains "
            "secondary to instruction, which is consistent with whole-class explanation."
            + speech_extra
        )
    if state == "Transition":
        return (
            f"The classroom shows elevated movement energy ({movement:.2f}) with changing "
            "occupancy over time, indicating a transition or circulation period. This is a useful "
            "signal for hallway/classroom flow and supervision analytics."
        )
    if state == "Group Work":
        return (
            f"The clip shows moderate-to-high occupancy with visible movement energy ({movement:.2f}), "
            "which is consistent with collaborative activity or group work. Aggregate analytics can "
            "highlight active zones without identifying individual students."
        )
    if state == "Independent Work":
        return (
            f"The classroom shows stable occupancy with a stationary proxy of {stationary:.0%}, "
            "suggesting independent learning or seated work. Movement is low enough to support a "
            "quiet-work interpretation."
        )
    return (
        "The clip shows low detected activity during the analyzed window. This may indicate a quiet "
        "period, limited camera coverage, or a clip segment with few visible people."
    )


def summarize_analysis(
    vision_result: dict[str, Any] | None,
    audio_result: dict[str, Any] | None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    vision_result = vision_result or {}
    audio_result = audio_result or {}
    series = vision_result.get("series") or []

    counts = [float(item.get("person_count", 0)) for item in series]
    movement_values = [float(item.get("movement_energy", 0)) for item in series]
    stationary_values = [float(item.get("stationary_ratio", 0)) for item in series]
    transition_values = [float(item.get("transition_intensity", 0)) for item in series]

    average_occupancy = _average(counts)
    peak_occupancy = max(counts) if counts else 0.0
    movement_energy = _average(movement_values)
    stationary_ratio = _average(stationary_values)
    transition_intensity = _average(transition_values)
    words_per_minute = float(audio_result.get("words_per_minute") or 0)

    activity_index = min(100.0, (average_occupancy * 6.0) + (movement_energy * 180.0) + transition_intensity)
    engagement_score = min(
        100.0,
        (stationary_ratio * 45.0) + (average_occupancy * 8.0) + min(25.0, words_per_minute / 4.0),
    )
    teacher_movement_proxy = min(100.0, movement_energy * 200.0)

    summary = {
        "average_occupancy": _round(average_occupancy),
        "peak_occupancy": int(peak_occupancy),
        "movement_energy": _round(movement_energy, 3),
        "stationary_ratio": _round(stationary_ratio, 3),
        "transition_intensity": _round(transition_intensity, 1),
        "speaking_pace_wpm": _round(words_per_minute, 1),
        "word_count": int(audio_result.get("word_count") or 0),
        "speaking_time_seconds": _round(audio_result.get("speaking_time_seconds") or 0, 1),
        "pause_count": int(audio_result.get("pause_count") or 0),
        "pause_ratio": _round(audio_result.get("pause_ratio") or 0, 3),
        "question_count": int(audio_result.get("question_count") or 0),
        "talk_ratio": _round(audio_result.get("talk_ratio") or 0, 2),
        "vocabulary_diversity": _round(audio_result.get("vocabulary_diversity") or 0, 2),
        "keywords": list(audio_result.get("keywords") or [])[:6],
        "teacher_movement_proxy": _round(teacher_movement_proxy, 1),
        "engagement_score": _round(engagement_score, 1),
        "activity_index": _round(activity_index, 1),
    }
    summary["classroom_state"] = classify_classroom_state(
        average_occupancy=average_occupancy,
        movement_energy=movement_energy,
        stationary_ratio=stationary_ratio,
        words_per_minute=words_per_minute,
        scenario_id=scenario_id,
    )
    summary["ai_insight_summary"] = build_ai_insight_summary(summary)
    return summary
