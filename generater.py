"""
Eğitilmiş QraXAi modeliyle (train.py çıktısı olan klasör) metin üretir.

Kullanım:
    python generater.py "Once upon a time, there was a little girl"
    python generater.py "Once upon a time" -n 3 --temperature 0.9 --stream
    python generater.py            # interaktif mod
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

DEFAULT_MODEL_DIR = "qraxai"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eğitilmiş QraXAi modeliyle metin üretir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Başlangıç metni (verilmezse interaktif mod açılır)",
    )
    parser.add_argument(
        "-m", "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="train.py çıktısındaki model klasörü",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1,
        help="Kaç farklı metin üretilsin",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Üretilecek en fazla token sayısı",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0 verirsen greedy (her adımda en olası token)",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="1.0 = kapalı",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Tekrarlanabilir üretim için rastgelelik tohumu",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Token'ları üretilirken canlı yazdır",
    )
    return parser.parse_args()


def pick_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir, device):
    path = Path(model_dir)
    if not path.exists():
        sys.exit(
            f"Model klasörü bulunamadı: {path.resolve()}\n"
            f"Önce 'python train.py' ile modeli eğitin."
        )

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        load_kwargs["dtype"] = torch.bfloat16  # GPU'da bellek/hız için bf16
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model = model.to(device).eval()
    return tokenizer, model


def generate_text(model, tokenizer, prompt, args, device, seed=None):
    max_seq_len = model.config.max_seq_len
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Bağlam penceresi: prompt + yeni token'lar max_seq_len'i aşmamalı,
    # çünkü model uzun dizilerde ValueError veriyor.
    if input_ids.size(1) > max_seq_len - 1:
        input_ids = input_ids[:, -(max_seq_len - 1):]
        print(
            f"[uyarı] Prompt bağlama sığmadı, son {input_ids.size(1)} token kullanılıyor.",
            file=sys.stderr,
        )

    max_new_tokens = min(args.max_new_tokens, max_seq_len - input_ids.size(1))
    if max_new_tokens < args.max_new_tokens:
        print(
            f"[bilgi] Bağlam {max_seq_len} token; max_new_tokens "
            f"{max_new_tokens} ile sınırlandı.",
            file=sys.stderr,
        )

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": max_new_tokens,
        # GPT-2'de eos = <|endoftext|>: hikâye bitince üretim durur.
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        kwargs.update(do_sample=False)

    if args.stream:
        kwargs["streamer"] = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    if seed is not None:
        torch.manual_seed(seed)

    with torch.inference_mode():
        output_ids = model.generate(**kwargs)

    if args.stream:
        print()  # streamer satırı bitirmemiş olabilir

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_samples(model, tokenizer, prompt, args, device):
    for i in range(args.num_samples):
        seed = None if args.seed is None else args.seed + i
        if args.num_samples > 1:
            print(f"\n--- Örnek {i + 1}/{args.num_samples} ---")
        text = generate_text(model, tokenizer, prompt, args, device, seed)
        if not args.stream:
            print(text, end="\n\n" if args.num_samples > 1 else "\n")


def interactive(model, tokenizer, args, device):
    print("İnteraktif mod. Çıkmak için 'q' yaz veya Ctrl-D.")
    sample_no = 0
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"q", "quit", "exit"}:
            break
        if not prompt.strip():
            continue

        for i in range(args.num_samples):
            sample_no += 1
            seed = None if args.seed is None else args.seed + sample_no
            if args.num_samples > 1:
                print(f"\n--- Örnek {i + 1}/{args.num_samples} ---")
            try:
                text = generate_text(model, tokenizer, prompt, args, device, seed)
            except KeyboardInterrupt:
                print("\n[üretim iptal edildi]")
                break
            if not args.stream:
                print(text)


def main():
    args = parse_args()

    if args.num_samples < 1:
        sys.exit("--num-samples en az 1 olmalı.")
    if args.max_new_tokens < 1:
        sys.exit("--max-new-tokens en az 1 olmalı.")

    device = pick_device(args.device)
    tokenizer, model = load_model(args.model_dir, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {Path(args.model_dir).resolve()} | Cihaz: {device} | "
        f"Parametre: {n_params:,} | Bağlam: {model.config.max_seq_len}"
    )
    if args.temperature > 0:
        print(
            f"Üretim: temperature={args.temperature}, "
            f"top_k={args.top_k}, top_p={args.top_p}, "
            f"repetition_penalty={args.repetition_penalty}"
        )
    else:
        print("Üretim: greedy (temperature=0)")

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        interactive(model, tokenizer, args, device)
        return

    run_samples(model, tokenizer, prompt, args, device)


if __name__ == "__main__":
    main()
