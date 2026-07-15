#!/usr/bin/env python3
"""
filter_transcripts_only.py

Strips a channel_data.txt file (produced by extract_channel_for_finetune.py)
down to just the transcript text of each video, dropping video IDs, titles,
upload dates, durations, descriptions, hashtags, and links.

USAGE:
    python filter_transcripts_only.py channel_data.txt --output transcripts_only.txt
"""

import argparse
import re
from pathlib import Path


def extract_transcripts(text):
    blocks = text.split("=" * 80)
    transcripts = []
    for block in blocks:
        match = re.search(r"TRANSCRIPT:\n(.*)", block, re.DOTALL)
        if match:
            transcript = match.group(1).strip()
            if transcript:
                transcripts.append(transcript)
    return transcripts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to channel_data.txt")
    parser.add_argument("--output", default="transcripts_only.txt", help="Output text file path")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    transcripts = extract_transcripts(text)

    out_path = Path(args.output)
    out_path.write_text("\n\n".join(transcripts), encoding="utf-8")

    print(f"Wrote {len(transcripts)} transcripts to {out_path.resolve()}")


if __name__ == "__main__":
    main()