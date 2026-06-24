"""Build a narrated 1080p business demo video for Classroom Intelligence.

Business-first cut aimed at a commercial / hardware partner: opens with the
opportunity, translates every metric into an outcome (flow & safety, space use,
instructional balance, room-level state), positions privacy as a strength, and
closes with a subtle pilot invitation. Reuses the existing annotated footage and
metrics (nothing is recomputed); adds a neural AI voiceover (edge-tts) and
branded slides, then stitches with the bundled ffmpeg.

Usage:
    .venv/Scripts/python.exe scripts/build_demo_video.py
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo_video"
AUDIO = OUT / "audio"
SLIDES = OUT / "slides"
SEGS = OUT / "segments"
for d in (OUT, AUDIO, SLIDES, SEGS):
    d.mkdir(parents=True, exist_ok=True)

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1920, 1080
VOICE = "en-US-GuyNeural"
RATE = "-6%"          # a touch slower for a confident, salesy cadence
MIN_FOOTAGE = 11.0    # keep the live analysis on screen longer

INK = (19, 34, 45)
MUTED = (97, 113, 125)
LINE = (216, 225, 230)
TEAL = (31, 122, 109)
AMBER = (201, 129, 32)
BLUE = (36, 82, 122)
PLUM = (123, 90, 160)
WASH = (238, 245, 243)
WHITE = (255, 255, 255)
PAD_BG = "0x0f1b24"


# ---------------------------------------------------------------- fonts / text
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    families = {
        "regular": ["segoeui.ttf", "arial.ttf"],
        "semibold": ["segoeuisb.ttf", "segoeuib.ttf", "arialbd.ttf"],
        "bold": ["segoeuib.ttf", "arialbd.ttf"],
        "light": ["segoeuil.ttf", "segoeui.ttf", "arial.ttf"],
    }
    for name in families.get(weight, families["regular"]):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def gradient_bg():
    img = Image.new("RGB", (W, H), WHITE)
    for y in range(H):
        t = y / H
        img.paste(tuple(int(WASH[i] + (WHITE[i] - WASH[i]) * t) for i in range(3)), (0, y, W, y + 1))
    return img


def footer(d):
    d.line([(120, H - 92), (W - 120, H - 92)], fill=LINE, width=2)
    d.text((120, H - 72), "Classroom Intelligence", font=font(26, "semibold"), fill=INK)
    d.text((W - 120, H - 72),
           "Anonymous aggregate analytics  ·  no identity recognition",
           font=font(24), fill=MUTED, anchor="ra")


def pill(d, x, y, text, fg, bg):
    fnt = font(26, "semibold")
    w = int(d.textlength(text, font=fnt) + 48)
    d.rounded_rectangle([x, y, x + w, y + 50], radius=25, fill=bg)
    d.text((x + 24, y + 11), text, font=fnt, fill=fg)
    return x + w + 16


def kpi_card(d, x, y, w, h, label, value, accent):
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=WHITE, outline=LINE, width=2)
    d.rounded_rectangle([x, y, x + w, y + 8], radius=4, fill=accent)
    d.text((x + 26, y + 30), label.upper(), font=font(23, "semibold"), fill=MUTED)
    val, fnt = str(value), font(60, "bold")
    while d.textlength(val, font=fnt) > w - 52 and fnt.size > 28:
        fnt = font(fnt.size - 4, "bold")
    d.text((x + 26, y + 70), val, font=fnt, fill=INK)


def insight(d, x, y, w, text, tone=TEAL, bg=(231, 243, 239), br=(189, 219, 211)):
    fnt = font(30)
    lines = wrap(d, text, fnt, w - 80)
    h = 44 + len(lines) * 42
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=bg, outline=br, width=2)
    d.rounded_rectangle([x, y, x + 8, y + h], radius=3, fill=tone)
    for i, ln in enumerate(lines):
        d.text((x + 36, y + 22 + i * 42), ln, font=fnt, fill=INK)
    return h


# ---------------------------------------------------------------- slide types
def slide_statement(path, eyebrow, lines, sub=None, accent=TEAL):
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    total = len(lines) * 92 + (70 if sub else 0)
    y = (H - total) // 2 - 30
    d.rounded_rectangle([120, y - 60, 210, y - 50], radius=5, fill=accent)
    if eyebrow:
        d.text((120, y - 38), eyebrow.upper(), font=font(30, "bold"), fill=accent)
    for ln in lines:
        d.text((120, y), ln, font=font(76, "bold"), fill=INK)
        y += 92
    if sub:
        y += 14
        for ln in wrap(d, sub, font(36), W - 480):
            d.text((120, y), ln, font=font(36), fill=MUTED)
            y += 50
    footer(d)
    img.save(path)


def slide_platform(path):
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([120, 250, 210, 260], radius=5, fill=TEAL)
    d.text((120, 285), "Classroom Intelligence", font=font(84, "bold"), fill=INK)
    tag = ("Turn the classroom video you already capture into safe, aggregate insight — "
           "occupancy, movement, and instruction, at a glance.")
    for i, ln in enumerate(wrap(d, tag, font(36), W - 520)):
        d.text((120, 405 + i * 52), ln, font=font(36), fill=MUTED)
    x = 120
    x = pill(d, x, 560, "Vision + speech analytics", BLUE, (231, 238, 244))
    x = pill(d, x, 560, "Privacy by design", TEAL, (228, 242, 238))
    x = pill(d, x, 560, "Runs on standard hardware", INK, (231, 235, 237))
    d.text((120, 660), "No faces  ·  no student IDs  ·  no emotion detection",
           font=font(30, "semibold"), fill=INK)
    footer(d)
    img.save(path)


def slide_clip(path, outcome, title, cards, why):
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    d.text((120, 108), outcome.upper(), font=font(30, "bold"), fill=TEAL)
    d.text((120, 152), title, font=font(64, "bold"), fill=INK)
    n, gap = len(cards), 28
    cw = (W - 240 - gap * (n - 1)) // n
    ch, y = 200, 290
    accents = [TEAL, BLUE, AMBER, PLUM]
    for i, (label, value) in enumerate(cards):
        kpi_card(d, 120 + i * (cw + gap), y, cw, ch, label, value, accents[i % 4])
    insight(d, 120, y + ch + 46, W - 240, why)
    footer(d)
    img.save(path)


def slide_feature(path, eyebrow, title, body, pills=None, tone=TEAL):
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    d.text((120, 150), eyebrow.upper(), font=font(30, "bold"), fill=tone)
    d.text((120, 196), title, font=font(68, "bold"), fill=INK)
    y = 330
    if pills:
        x = 120
        for label, fg, bg in pills:
            x = pill(d, x, y, label, fg, bg)
        y += 96
    for ln in wrap(d, body, font(38), W - 320):
        d.text((120, y), ln, font=font(38), fill=MUTED)
        y += 56
    footer(d)
    img.save(path)


def lower_third(path, title, stat):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    by = H - 235
    d.rounded_rectangle([90, by, 90 + 760, by + 132], radius=18, fill=(15, 27, 36, 224))
    d.rounded_rectangle([90, by, 98, by + 132], radius=4, fill=TEAL + (255,))
    d.text((128, by + 22), title, font=font(40, "bold"), fill=(255, 255, 255, 255))
    d.text((128, by + 80), stat, font=font(28), fill=(196, 214, 210, 255))
    d.rounded_rectangle([90, by - 56, 470, by - 8], radius=24, fill=(31, 122, 109, 235))
    d.text((112, by - 47), "Anonymous tracking · IDs only", font=font(24, "semibold"),
           fill=(255, 255, 255, 255))
    img.save(path)


# ---------------------------------------------------------------- encode
def media_duration(path):
    out = subprocess.run([FF, "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not m:
        raise RuntimeError(f"no duration for {path}")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


COMMON_V = ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-r", "30", "-g", "60"]
COMMON_A = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise SystemExit(f"ffmpeg failed: {' '.join(cmd[:6])} ...")


def seg_slide(img, audio, out, tail=0.8):
    dur = media_duration(audio) + tail
    run([FF, "-y", "-loop", "1", "-i", str(img), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-t", f"{dur:.3f}",
         "-vf", "scale=1920:1080,setsar=1,format=yuv420p", "-af", "apad",
         *COMMON_V, *COMMON_A, "-movflags", "+faststart", str(out)])


def seg_footage(video, audio, overlay, out, start=2.0, tail=0.6, min_dur=MIN_FOOTAGE):
    dur = max(media_duration(audio) + tail, min_dur)
    fc = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
          f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={PAD_BG},setsar=1,fps=30[bg];"
          f"[bg][2:v]overlay=0:0,format=yuv420p[v]")
    run([FF, "-y", "-ss", str(start), "-i", str(video), "-i", str(audio), "-i", str(overlay),
         "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
         "-t", f"{dur:.3f}", "-af", "apad",
         *COMMON_V, *COMMON_A, "-movflags", "+faststart", str(out)])


# ---------------------------------------------------------------- data
def load_clips():
    by_media = {}
    for f in glob.glob(str(ROOT / "outputs" / "metrics" / "*_metrics.json")):
        d = json.load(open(f, encoding="utf-8"))
        name, mt = d.get("media", {}).get("name"), os.path.getmtime(f)
        if name and (name not in by_media or mt > by_media[name][0]):
            by_media[name] = (mt, d)
    return {k: v[1] for k, v in by_media.items()}


def main():
    clips = load_clips()
    order = ["Task1_Kids_Walking.webm", "Task2_4_CCTV_Chairs_Moving.mkv",
             "Task3_Teacher_Speaking.webm", "Task5_Aggregate_Classroom.mkv"]
    missing = [c for c in order if c not in clips]
    if missing:
        raise SystemExit(f"Missing analysis for: {missing}. Run scripts/run_validation.py first.")

    s = lambda n: clips[n]["summary"]
    av = lambda n: Path(clips[n]["vision"]["annotated_video_path"])
    s1, s2, s3, s4 = s(order[0]), s(order[1]), s(order[2]), s(order[3])

    plan = [
        # --- opening: opportunity, then the platform ---
        dict(key="00_hook", kind="slide",
             narration=("Classrooms everywhere are full of cameras and microphones. But almost "
                        "none of that footage ever becomes insight."),
             render=lambda p: slide_statement(p, "The opportunity",
                        ["Hours of classroom video.", "Almost none of it", "becomes insight."])),
        dict(key="01_platform", kind="slide",
             narration=("Classroom Intelligence changes that — turning the video you already "
                        "capture into safe, aggregate understanding of how a classroom is "
                        "actually being used."),
             render=lambda p: slide_platform(p)),

        # --- scenario 1: flow & safety ---
        dict(key="02_c1_slide", kind="slide",
             narration=(f"Take a transition period. The platform tracks how many people are "
                        f"present and how much they are moving — here, about "
                        f"{round(s1['average_occupancy'])} on average, peaking at "
                        f"{s1['peak_occupancy']}. For a busy room, that is a live read on flow, "
                        f"supervision, and safety."),
             render=lambda p: slide_clip(p, "Flow · Supervision · Safety", "Movement & Transitions",
                        [("Avg Occupancy", s1['average_occupancy']), ("Peak Occupancy", s1['peak_occupancy']),
                         ("Movement", s1['movement_energy']), ("Transition", s1['transition_intensity'])],
                        "Live counts and movement during transitions give a real-time read on flow, supervision, and safety.")),
        dict(key="02_c1_footage", kind="footage", video=av(order[0]), start=3.0,
             lt=("Movement & Transitions", f"Avg occupancy {round(s1['average_occupancy'])} · Peak {s1['peak_occupancy']}"),
             narration=("Every person becomes an anonymous box with a tracking ID — counts and "
                        "motion, never identities.")),

        # --- scenario 2: space use & engagement ---
        dict(key="03_c2_slide", kind="slide",
             narration=(f"During independent work, occupancy stays high — around "
                        f"{round(s2['average_occupancy'])} — while movement drops and the "
                        f"stationary signal climbs to {s2['stationary_ratio']*100:.0f} percent. "
                        f"That tells you a room is in active use, and how settled it is."),
             render=lambda p: slide_clip(p, "Space use · Engagement", "Seated, Independent Work",
                        [("Avg Occupancy", s2['average_occupancy']), ("Stationary", f"{s2['stationary_ratio']*100:.0f}%"),
                         ("Movement", s2['movement_energy']), ("State", s2['classroom_state'])],
                        "High, settled occupancy shows a room in active use — a practical signal for space planning and engagement.")),
        dict(key="03_c2_footage", kind="footage", video=av(order[1]), start=2.0,
             lt=("Seated, Independent Work", f"Stationary {s2['stationary_ratio']*100:.0f}% · {s2['classroom_state']}"),
             narration=("The same anonymous tracking shows a calm, fully-used room, with students "
                        "settled at their work.")),

        # --- scenario 3: instructional balance ---
        dict(key="04_c3_slide", kind="slide",
             narration=(f"Add speech, and a new dimension opens up. The platform measured a "
                        f"speaking pace near {round(s3['speaking_pace_wpm'])} words a minute and "
                        f"almost {round(s3['word_count'], -1)} words — a window into teacher-talk "
                        f"versus student-work time, one of the most requested signals in education."),
             render=lambda p: slide_clip(p, "Instructional Balance", "Teacher Speaking",
                        [("Speaking Pace", f"{s3['speaking_pace_wpm']} WPM"), ("Word Count", s3['word_count']),
                         ("Avg Occupancy", s3['average_occupancy']), ("State", s3['classroom_state'])],
                        "Teacher-talk versus student-work time — one of the most requested signals in education — without identifying anyone.")),
        dict(key="04_c3_footage", kind="footage", video=av(order[2]), start=2.0,
             lt=("Teacher Speaking", f"{s3['speaking_pace_wpm']} WPM · {s3['word_count']} words"),
             narration=("Cadence, word count, and a full transcript — captured without ever "
                        "identifying who is speaking.")),

        # --- scenario 4: whole room ---
        dict(key="05_c4_slide", kind="slide",
             narration=(f"And it all comes together. Occupancy — averaging about "
                        f"{round(s4['average_occupancy'])}, peaking at {s4['peak_occupancy']} — "
                        f"movement, and stillness combine into a single, room-level picture a "
                        f"school leader could scan across every classroom."),
             render=lambda p: slide_clip(p, "The whole room at a glance", "Aggregate Classroom Dynamics",
                        [("Avg Occupancy", s4['average_occupancy']), ("Peak Occupancy", s4['peak_occupancy']),
                         ("Movement", s4['movement_energy']), ("Transition", s4['transition_intensity'])],
                        "Occupancy, movement, and stillness combine into one room-level picture a leader can scan at a glance.")),
        dict(key="05_c4_footage", kind="footage", video=av(order[3]), start=3.0,
             lt=("Aggregate Classroom Dynamics", f"Avg occupancy {round(s4['average_occupancy'])} · Peak {s4['peak_occupancy']}"),
             narration=("One model, one camera feed, turned into a live read on the entire room.")),

        # --- decisions ---
        dict(key="06_decisions", kind="slide",
             narration=("Everything lands in a simple dashboard and exports cleanly to the tools "
                        "schools already use — spreadsheets, data feeds, and shareable reports. "
                        "No new workflow to learn."),
             render=lambda p: slide_feature(p, "From footage to decisions", "Insight that fits existing systems",
                        "It plugs into the tools schools and operators already run — nothing to rip and replace.",
                        pills=[("JSON", BLUE, (231, 238, 244)), ("CSV", TEAL, (228, 242, 238)),
                               ("Markdown", INK, (231, 235, 237)), ("HTML report", AMBER, (250, 241, 228))])),

        # --- privacy as strength ---
        dict(key="07_privacy", kind="slide",
             narration=("And it is private by design. No face recognition. No student "
                        "identification. No emotion detection. Only anonymous, aggregate "
                        "analytics — safe for children, and defensible to parents and regulators."),
             render=lambda p: slide_feature(p, "Privacy by design", "Built for a children's setting",
                        "Anonymous, aggregate analytics only — defensible to parents, schools, and regulators.",
                        pills=[("No face recognition", INK, (231, 235, 237)), ("No student IDs", INK, (231, 235, 237)),
                               ("No emotion detection", INK, (231, 235, 237))])),

        # --- scale (subtle partner signal) ---
        dict(key="08_scale", kind="slide",
             narration=("It runs on standard hardware and deploys today — and it is built to "
                        "grow. From a single classroom to an entire campus: every room on one "
                        "screen, patterns tracked over time, and signals tuned to what matters "
                        "most. The more cameras and context it sees, the sharper the picture "
                        "becomes."),
             render=lambda p: slide_feature(p, "Built to scale", "One classroom today. A campus tomorrow.",
                        "Deploys on standard hardware now — and grows to every room on one screen, with trends over time and custom signals. The more it sees, the sharper the picture.",
                        pills=[("Standard hardware", TEAL, (228, 242, 238)), ("Many cameras · multi-site", BLUE, (231, 238, 244)),
                               ("Trends & custom KPIs", INK, (231, 235, 237))], tone=BLUE)),

        # --- subtle close ---
        dict(key="09_close", kind="slide",
             narration=("Classroom Intelligence is working today, and ready to prove itself in "
                        "real classrooms. Let's explore what it could show in yours."),
             render=lambda p: slide_statement(p, "Working today",
                        ["Ready for the", "real classroom."],
                        sub="Let's explore what Classroom Intelligence could show in yours.", accent=TEAL)),
    ]

    print("[1/4] voiceover ...")
    async def synth():
        for it in plan:
            mp3 = AUDIO / f"{it['key']}.mp3"
            await edge_tts.Communicate(it["narration"], VOICE, rate=RATE).save(str(mp3))
            it["audio"] = mp3
            print("   voice:", it["key"])
    asyncio.run(synth())

    print("[2/4] slides ...")
    for it in plan:
        if it["kind"] == "slide":
            png = SLIDES / f"{it['key']}.png"
            it["render"](png)
            it["img"] = png
        else:
            ov = SLIDES / f"{it['key']}_lt.png"
            lower_third(ov, it["lt"][0], it["lt"][1])
            it["overlay"] = ov
        print("   slide:", it["key"])

    print("[3/4] segments ...")
    seg_paths = []
    for it in plan:
        seg = SEGS / f"{it['key']}.mp4"
        if it["kind"] == "slide":
            seg_slide(it["img"], it["audio"], seg)
        else:
            seg_footage(it["video"], it["audio"], it["overlay"], seg, start=it.get("start", 2.0))
        seg_paths.append(seg)
        print("   segment:", it["key"])

    print("[4/4] stitching ...")
    listfile = OUT / "concat.txt"
    listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in seg_paths), encoding="utf-8")
    final = OUT / "Classroom_Intelligence_Demo.mp4"
    run([FF, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", "-movflags", "+faststart", str(final)])
    print(f"\nDONE -> {final}  ({media_duration(final):.1f}s, {final.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
