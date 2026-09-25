# QraXAi

A tiny GPT-style language model written **from scratch** in PyTorch and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-red.svg)](https://pytorch.org/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-QraXAi--Basic--45M-blue)](https://huggingface.co/coderian/QraXAi-Basic-45M)

QraXAi is a decoder-only Transformer (~44.75M parameters) implemented by hand — no
`transformers` model classes, no GPT-2 weights — and trained from scratch on the
TinyStories dataset using the GPT-2 BPE tokenizer (only the tokenizer is shared with
GPT-2, the architecture and weights are custom).

This repository contains the full pipeline: dataset download, training, text generation
and upload to the Hugging Face Hub.

**Trained model:** [`coderian/QraXAi-Basic-45M`](https://huggingface.co/coderian/QraXAi-Basic-45M)

> Experimental research model, intended for learning and demo purposes.

## Highlights

- Hand-written GPT architecture (`model.py` + `configuration_qraxai.py`) with pre-norm
  blocks, causal self-attention and GELU feed-forward layers.
- Full training loop with bf16 mixed precision on CUDA, gradient accumulation,
  gradient clipping and a live `tqdm` progress bar.
- Saves a ready-to-load Hugging Face model folder (`AutoModelForCausalLM` with
  `trust_remote_code=True`).
- CLI generation script with sampling controls, streaming output and an interactive mode.
- One-command upload to the Hugging Face Hub.

## Model architecture

| | |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style), pre-norm |
| Parameters | **44,751,872** (~44.75M), all trainable |
| Layers | 24 |
| Hidden size | 256 |
| Attention heads | 8 (head dim 32) |
| Feed-forward | 4x hidden, GELU |
| Context length | 256 tokens (hard limit) |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Position encoding | Learned absolute embeddings |
| Normalization | LayerNorm |
| Weight tying | No (`lm_head` is separate) |
| KV cache | No — generation recomputes the full context at every step |
| Weights | fp32, 179 MB (`model.safetensors`) |
| Special tokens | `bos = eos = <\|endoftext\|>` (id 50256) |
| Custom code | Yes — requires `trust_remote_code=True` |

## Repository structure

```
QraXAi-2/
├── configuration_qraxai.py   # GPTConfig — custom Hugging Face config
├── model.py                  # QraXAiForCausalLM — hand-written GPT model
├── train.py                  # Training script (tokenize -> train -> save)
├── generater.py              # CLI text generation
├── push.py                   # Upload the trained model to the Hugging Face Hub
├── requirements.txt          # Python dependencies
├── dataset/
│   └── download.py           # Streams TinyStories into dataset/data.txt
└── LICENSE
```

Generated at runtime (git-ignored):

| Path | Description |
|---|---|
| `dataset/data.txt` | Training text (~111 MB, 130k TinyStories) |
| `qraxai/` | Trained model folder (config, weights, tokenizer, custom code) — uploaded to the HF Hub by `push.py` |

## Installation

Python 3.9+ is recommended (developed with 3.11). A CUDA-capable GPU is optional —
training and generation also run on CPU (slower, fp32).

```bash
git clone https://github.com/coderian/QraXAi-2.git
cd QraXAi-2

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies: PyTorch, Transformers, Datasets, huggingface_hub and tqdm. For GPU training,
install the CUDA build of PyTorch matching your CUDA version
(see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Usage

### 1. Download the dataset

Streams the first 130,000 stories of `roneneldan/TinyStories` into `dataset/data.txt`:

```bash
python dataset/download.py
```

### 2. Train

```bash
python train.py
```

The script tokenizes the text with the GPT-2 tokenizer, trains the model and writes a
Hugging Face-compatible folder to `qraxai/` (`config.json`, `model.safetensors`,
tokenizer files and the custom code needed for `trust_remote_code=True`).

Training hyperparameters (`BLOCK_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`,
`N_LAYERS`, `EMBED_DIM`, ...) are defined at the top of `train.py`. CUDA with bf16 is
used automatically when available; otherwise training runs on CPU in fp32.

### 3. Generate text

```bash
# single sample
python generater.py "Once upon a time, there was a little girl"

# 3 samples, higher temperature, stream tokens as they are produced
python generater.py "Once upon a time" -n 3 --temperature 0.9 --stream

# greedy decoding
python generater.py "Once upon a time" --temperature 0

# interactive mode
python generater.py
```

Useful options: `--max-new-tokens`, `--top-k`, `--top-p`, `--repetition-penalty`,
`--seed`, `--device {auto,cuda,cpu}`, `-m/--model-dir`.

Because the context window is a hard limit of 256 tokens, the script truncates long
prompts and caps `max_new_tokens` so that `prompt + new tokens <= 256`.

### 4. Upload to the Hugging Face Hub

```bash
hf auth login        # once
python push.py
```

The target repository is set with `REPO_ID` in `push.py`.

## Training details

| | |
|---|---|
| Dataset | `roneneldan/TinyStories` (train split, streaming), first 130,000 stories |
| Data size | 115.7M characters -> 28.76M tokens -> ~112,350 training blocks |
| Objective | Next-token prediction (causal LM), cross-entropy |
| Epochs | 1 (~7,000 optimizer steps) |
| Batch size | 16 |
| Block size | 256 |
| Optimizer | AdamW, lr 3e-4 |
| Gradient clipping | 1.0 |
| Mixed precision | bf16 (on CUDA) |
| Seed | 42 |

## Example output

Prompt: `Once upon a time, there was a little girl named Lily`
(`temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1`):

> Once upon a time, there was a little girl named Lily who loved to play in the big,
> green field. One day, she found a shiny stone on top of her backyard. She picked it up
> and showed it to her mom.
>
> "Look mommy, I found a pretty mineral!" said Lily excitedly. "It's very pretty!"
>
> Her mom smiled and said, "That's right, sweetie..."

The full sample and model card are available on the
[Hugging Face model page](https://huggingface.co/coderian/QraXAi-Basic-45M).

## Limitations

- English only; trained exclusively on synthetic children's stories, so it knows little
  about the real world.
- Trained for a single epoch: grammar is mostly coherent, but content can be repetitive,
  inconsistent or nonsensical.
- Hard 256-token context; no sliding window, so long inputs must be truncated.
- No KV cache: generation cost grows quickly with sequence length.
- Not instruction-tuned — it does not follow instructions and is not a chat model.
- No safety filtering or alignment of any kind. Do not use in production or for
  user-facing applications.

## Acknowledgements

- Dataset and idea: **TinyStories: How Small Can Language Models Be and Still Speak
  Coherent English?** — Ronen Eldan and Yuanzhi Li, 2023
  ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)).
- Tokenizer: GPT-2 BPE (`gpt2`).
- Built with [PyTorch](https://pytorch.org/) and
  [Hugging Face Transformers](https://huggingface.co/docs/transformers).

## License

Released under the [MIT License](LICENSE). Note that the training dataset
([TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)) has its own
terms; check them if you plan to redistribute the data or use the model commercially.
