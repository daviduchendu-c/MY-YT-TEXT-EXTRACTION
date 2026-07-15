#!/usr/bin/env python3
"""
chat.py

Interactive CLI chat against the locally-running fine-tuned model, served by
Ollama (after `ollama create laurin -f Modelfile`).

USAGE:
    python chat.py [--model laurin] [--host http://localhost:11434]
"""

import argparse
import json
import sys

import requests


def stream_chat(host, model, messages):
    response = requests.post(
        f"{host}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
    )
    response.raise_for_status()

    full_reply = []
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        content = event.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)
            full_reply.append(content)
        if event.get("done"):
            break
    print()
    return "".join(full_reply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="laurin", help="Ollama model name (from `ollama create`)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama server URL")
    args = parser.parse_args()

    print(f"Chatting with '{args.model}' via {args.host}. Type 'exit' or Ctrl+C to quit.\n")

    messages = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        print(f"{args.model}: ", end="", flush=True)
        try:
            reply = stream_chat(args.host, args.model, messages)
        except requests.exceptions.ConnectionError:
            print(f"\nCould not reach Ollama at {args.host}. Is `ollama serve` running?", file=sys.stderr)
            break
        except requests.exceptions.HTTPError as e:
            print(f"\nOllama returned an error: {e}", file=sys.stderr)
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
