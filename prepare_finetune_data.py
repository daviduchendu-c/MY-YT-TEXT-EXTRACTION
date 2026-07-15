#!/usr/bin/env python3
"""
prepare_finetune_data.py

Converts channel_data.txt (video blocks with TITLE + TRANSCRIPT, produced by
extract_channel_for_finetune.py) into a chat-format JSONL dataset suitable for
QLoRA fine-tuning (e.g. via Unsloth on Google Colab).

Since the transcripts are pure monologue (no natural Q&A), each video's TITLE
is turned into a stand-in user prompt and its TRANSCRIPT becomes the assistant
reply, wrapped in a fixed persona system prompt. Transcripts longer than the
word-count budget are split into multiple examples (same prompt, successive
chunks) so no single training example exceeds the training sequence length.

USAGE:
    python prepare_finetune_data.py channel_data.txt \
        --train-output finetune_train.jsonl \
        --val-output finetune_val.jsonl
"""

import argparse
import json
import re
from pathlib import Path

PERSONA_SYSTEM_PROMPT = (
    "You are a direct, no-nonsense relationship and dating coach who speaks "
    "candidly to men about modern relationships, dating, and marriage. You "
    "back up your points with references to evolutionary psychology and "
    "biology, break advice down into clear numbered points, speak with blunt "
    "confidence, address the listener as \"guys\", and close out with a "
    "motivational, no-excuses tone."
)

# ~1500 tokens of transcript per training example, approximated as words.
MAX_WORDS_PER_CHUNK = 1100

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_title(title):
    title = EMOJI_PATTERN.sub("", title)
    title = re.sub(r"\s+", " ", title).strip(" -|:")
    return title


def extract_examples(text):
    blocks = text.split("=" * 80)
    videos = []
    for block in blocks:
        title_match = re.search(r"TITLE:\s*(.+)", block)
        transcript_match = re.search(r"TRANSCRIPT:\n(.*)", block, re.DOTALL)
        if not title_match or not transcript_match:
            continue
        title = title_match.group(1).strip()
        transcript = transcript_match.group(1).strip()
        if not title or not transcript or transcript == "[no transcript available]":
            continue
        videos.append((title, transcript))
    return videos


def chunk_transcript(transcript, max_words):
    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    chunks = []
    current = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_examples(videos):
    examples = []
    for title, transcript in videos:
        prompt = f"Give me your take on: {clean_title(title)}"
        for chunk in chunk_transcript(transcript, MAX_WORDS_PER_CHUNK):
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chunk},
                    ]
                }
            )
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to channel_data.txt")
    parser.add_argument("--train-output", default="finetune_train.jsonl")
    parser.add_argument("--val-output", default="finetune_val.jsonl")
    parser.add_argument("--val-every", type=int, default=20, help="Every Nth example goes to the validation split")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    videos = extract_examples(text)
    examples = build_examples(videos)

    train_path = Path(args.train_output)
    val_path = Path(args.val_output)

    word_counts = []
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for i, example in enumerate(examples):
            assistant_text = example["messages"][-1]["content"]
            word_counts.append(len(assistant_text.split()))
            line = json.dumps(example, ensure_ascii=False) + "\n"
            if (i + 1) % args.val_every == 0:
                val_f.write(line)
            else:
                train_f.write(line)

    n_val = sum(1 for i in range(len(examples)) if (i + 1) % args.val_every == 0)
    n_train = len(examples) - n_val

    print(f"Parsed {len(videos)} videos from {args.input}")
    print(f"Wrote {n_train} training examples to {train_path.resolve()}")
    print(f"Wrote {n_val} validation examples to {val_path.resolve()}")
    if word_counts:
        print(
            f"Assistant-reply word counts: min={min(word_counts)} "
            f"max={max(word_counts)} avg={sum(word_counts) / len(word_counts):.0f}"
        )


if __name__ == "__main__":
    main()
