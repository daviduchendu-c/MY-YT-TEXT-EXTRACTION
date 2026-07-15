#!/usr/bin/env python3
"""
extract_channel_for_finetune.py

Pulls a YouTube channel's video metadata + transcripts and writes them
out as a single clean text file, suitable for feeding into a fine-tuning
pipeline.

REQUIREMENTS (run these on your own machine first):
    pip install yt-dlp

USAGE:
    python extract_channel_for_finetune.py "https://www.youtube.com/@FlourishwithLaurin" \
        --output channel_data.txt \
        --max-videos 200 \
        --allow-auto-captions

NOTES:
- No API key required — yt-dlp scrapes public metadata + caption tracks directly.
- If a video has no manual captions, auto-generated captions will be used
  as a fallback (since you opted in), but auto-captions are lower quality
  (no punctuation, occasional mis-transcription) — worth spot-checking.
- Be mindful of copyright: transcripts are the channel owner's content.
  Using them to fine-tune a private/personal model is generally fine;
  redistributing the raw text or a model trained to reproduce it verbatim
  is a separate legal question you should think through, especially if
  it's not your own channel.
- Large channels = long runtime. --max-videos lets you cap it while testing.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    """Run a subprocess command and return stdout, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[warn] command failed: {' '.join(cmd)}\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def get_video_ids(channel_url, max_videos):
    """Use yt-dlp to list video IDs from a channel without downloading video files."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "%(id)s",
        channel_url,
    ]
    if max_videos:
        cmd += ["--playlist-end", str(max_videos)]
    out = run(cmd)
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_video_metadata_and_captions(video_id, allow_auto_captions, tmp_dir):
    """
    Fetch title/description/upload date via --dump-json,
    and download captions (manual, falling back to auto) as .vtt.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # --- metadata ---
    meta_out = run([sys.executable, "-m", "yt_dlp", "--dump-json", "--no-warnings", video_url])
    if not meta_out:
        return None
    try:
        meta = json.loads(meta_out.splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return None

    title = meta.get("title", "")
    description = meta.get("description", "")
    upload_date = meta.get("upload_date", "")
    duration = meta.get("duration", "")

    # --- captions ---
    sub_flags = ["--write-subs", "--sub-langs", "en.*"]
    if allow_auto_captions:
        sub_flags.append("--write-auto-subs")

    caption_prefix = tmp_dir / video_id
    run([
        sys.executable, "-m", "yt_dlp",
        "--skip-download",
        *sub_flags,
        "--sub-format", "vtt",
        "-o", str(caption_prefix) + ".%(ext)s",
        video_url,
    ])

    transcript_text = ""
    for vtt_file in tmp_dir.glob(f"{video_id}*.vtt"):
        transcript_text = vtt_to_text(vtt_file)
        vtt_file.unlink()
        break

    return {
        "id": video_id,
        "title": title,
        "description": description,
        "upload_date": upload_date,
        "duration": duration,
        "transcript": transcript_text,
    }


def vtt_to_text(vtt_path):
    """Strip timestamps/formatting from a .vtt file, return plain transcript text."""
    raw = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    text_lines = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        # strip inline tags like <00:00:01.000><c> word</c>
        line = re.sub(r"<[^>]+>", "", line)
        if line and line not in seen:
            text_lines.append(line)
            seen.add(line)

    return " ".join(text_lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("channel_url", help="Channel URL, e.g. https://www.youtube.com/@SomeChannel")
    parser.add_argument("--output", default="channel_data.txt", help="Output text file path")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap number of videos processed")
    parser.add_argument("--allow-auto-captions", action="store_true",
                         help="Fall back to auto-generated captions if no manual ones exist")
    args = parser.parse_args()

    tmp_dir = Path("./_yt_tmp")
    tmp_dir.mkdir(exist_ok=True)

    print(f"Fetching video list for {args.channel_url} ...")
    video_ids = get_video_ids(args.channel_url, args.max_videos)
    print(f"Found {len(video_ids)} videos.")

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for i, vid in enumerate(video_ids, 1):
            print(f"[{i}/{len(video_ids)}] {vid}")
            data = get_video_metadata_and_captions(vid, args.allow_auto_captions, tmp_dir)
            if not data:
                continue

            f.write("=" * 80 + "\n")
            f.write(f"VIDEO_ID: {data['id']}\n")
            f.write(f"TITLE: {data['title']}\n")
            f.write(f"UPLOAD_DATE: {data['upload_date']}\n")
            f.write(f"DURATION_SECONDS: {data['duration']}\n")
            f.write("-" * 80 + "\n")
            f.write("DESCRIPTION:\n")
            f.write(data["description"].strip() + "\n")
            f.write("-" * 80 + "\n")
            f.write("TRANSCRIPT:\n")
            f.write((data["transcript"] or "[no transcript available]") + "\n")
            f.write("\n")

    for leftover in tmp_dir.glob("*"):
        leftover.unlink()
    tmp_dir.rmdir()

    print(f"\nDone. Wrote data for {len(video_ids)} videos to {out_path.resolve()}")


if __name__ == "__main__":
    main()
