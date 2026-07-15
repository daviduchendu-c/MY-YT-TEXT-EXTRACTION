##Creator style chatbot

Turns the scraped channel transcripts into a small, locally-runnable model that talks in the
creator's voice — a direct, no-nonsense relationship/dating coach who addresses men, leans on
evolutionary-psych/biology framing, breaks advice into numbered points, and closes with a
motivational sign-off.

Pipeline: build a training dataset from the transcripts -> fine-tune a small open model for free
On Google Colab's GPU -> run the fine-tuned model locally with Ollama -> chat with it.

## Prerequisites

- Python 3.9+
- A Google account (for Google Colab, used for the free GPU fine-tuning step)
- [Ollama](https://ollama.com) installed locally
- `pip install -r requirements.txt` (installs `requests`, used by `chat.py`)

## Step 1 — Build the training dataset

```bash
python prepare_finetune_data.py channel_data.txt
```

This parses `channel_data.txt` (video blocks with `TITLE:` and `TRANSCRIPT:` fields) and writes:

- `finetune_train.jsonl` — training examples
- `finetune_val.jsonl` — held-out validation examples (every 20th example)

Each example pairs a video's title (turned into a stand-in prompt like `"Give me your take on:
..."`) with its transcript (the in-character reply), wrapped in a fixed persona system prompt.
Transcripts longer than ~1100 words are split into multiple chunks so no single example is too
long to train on.

Re-run this command any time `channel_data.txt` is refreshed with more scraped videos — it
regenerates both JSONL files from scratch.

## Step 2 — Fine-tune on Google Colab (free GPU)

1. Upload `finetune_colab.ipynb` to [Google Colab](https://colab.research.google.com) (File ->
   Upload notebook, or open it directly from Google Drive.
2. **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.**
3. Run the cells top to bottom:
   - Installs [Unsloth](https://github.com/unslothai/unsloth) for QLoRA fine-tuning.
   - Prompts you to upload `finetune_train.jsonl` and `finetune_val.jsonl` (from Step 1).
   - Loads `unsloth/Meta-Llama-3.1-8B-bnb-4bit` in 4-bit and attaches LoRA adapters.
   - Trains for a few epochs, then runs a smoke-test generation so you can eyeball whether the
     voice/style landed.
   - Exports the fine-tuned model to GGUF (`q4_k_m` quantization) and downloads it to your
     computer.
4. Save the downloaded `.gguf` file into this project folder.

This step is free but not instant — expect the training cell to take a while, depending on Colab's
GPU availability at the time.

## Step 3 — Load the model into Ollama

The `Modelfile` in this folder expects a file named `laurin-model.Q4_K_M.gguf` in the same
directory. Either rename the file you downloaded from Colab to match, or edit the `FROM` line in
`Modelfile` to point at the actual filename. Then:

```bash
ollama create laurin -f Modelfile
```

Quick manual check:

```bash
ollama run laurin "Give me your take on: dating someone who just got out of a long relationship"
```

## Step 4 — Chat with it

```bash
pip install -r requirements.txt
python chat.py
```

Example session:

```
Chatting with 'laurin' via http://localhost:11434. Type 'exit' or Ctrl+C to quit.

You: should I date a single mom?
laurin: ...
You: exit
```

Flags:

- `--model <name>` — Ollama model name if you created it under a different name (default: `laurin`)
- `--host <url>` — Ollama server URL (default: `http://localhost:11434`)

## Notes / troubleshooting

- **"Could not reach Ollama"** — make sure `ollama serve` is running (or just run `ollama run
  laurin` once first, which starts the server).
- **Changing the persona/tone** — the persona description lives in two places that must stay in
  sync: `PERSONA_SYSTEM_PROMPT` in `prepare_finetune_data.py` (baked into training data) and the
  `SYSTEM` block in `Modelfile` (used at inference time). Edit both, regenerate the dataset, and
  re-run fine-tuning for changes to fully take effect.
- **Adding more videos** — re-run `extract_channel_for_finetune.py` to refresh `channel_data.txt`,
  then repeat Steps 1–3.
