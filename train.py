"""
QraXAi modelini GPT-2 tokenizer ile eğitir ve AutoModel ile
(trust_remote_code=True) yüklenebilecek hazır bir model klasörü üretir.

Kullanım:
    python train.py
"""

import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast

from configuration_qraxai import GPTConfig
from model import QraXAiForCausalLM

# ---------------------------------------------------------------- ayarlar

DATA_FILE = "./dataset/data.txt"   # eğitim metni
OUTPUT_DIR = "qraxai"    # modelin kaydedileceği klasör

BLOCK_SIZE = 256         # bağlam uzunluğu (token)
TOKENIZE_CHUNK = 1_000_000  # tokenizasyon parça boyutu (karakter)
BATCH_SIZE = 16
GRAD_ACCUM = 1           # gradyan biriktirme adımı
LEARNING_RATE = 3e-4
EPOCHS = 1
LOG_INTERVAL = 50        # her kaç adımda bir kayıp yazdırılsın

N_LAYERS = 24
EMBED_DIM = 256
SEED = 42


def format_duration(seconds):
    """Saniyeyi okunabilir süreye çevirir: '1s 23dk 45sn' gibi."""
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}s {minutes}dk {secs}sn"
    return f"{minutes}dk {secs}sn"


def split_text(text, size):
    """Metni, `size` karakteri geçmeyecek parçalara satır satır böler."""
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > size:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return parts


class TextDataset(Dataset):
    """Metni token'lar ve sabit uzunlukta (block_size) parçalara böler."""

    def __init__(self, data_file, tokenizer, block_size):
        text = Path(data_file).read_text(encoding="utf-8")

        # Parçaları liste halinde tek çağrıda token'la: fast tokenizer
        # böylece çekirdekleri paralel kullanır.
        parts = split_text(text, TOKENIZE_CHUNK)
        token_ids = [
            token_id
            for part_ids in tokenizer(parts)["input_ids"]
            for token_id in part_ids
        ]

        # Tüm token'ları tek düz tensörde tut: Python listesine göre çok daha
        # az bellek kullanır ve __getitem__ içinde kopya oluşturmaz.
        self.block_size = block_size
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Her parça block_size + 1 token.
        self.num_chunks = (len(self.token_ids) - block_size) // block_size

        if self.num_chunks <= 0:
            raise ValueError(
                f"{data_file} çok kısa: en az {block_size + 1} token gerekli."
            )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = self.token_ids[start : start + self.block_size + 1]

        # model.forward hedefi kendi içinde kaydırır (shift_labels = labels[:, 1:])
        # bu yüzden input ve label aynı token dizisidir.
        inputs = chunk[:-1]
        return inputs, inputs.clone()  # input_ids, labels


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"  # bf16 karışık hassasiyet yalnızca GPU'da

    # GPT-2 tokenizer: vocab_size=50257, özel token <|endoftext|>
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Tüm metni tek seferde token'ladığımız için uzunluk uyarısını geçici kapat.
    model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    dataset = TextDataset(DATA_FILE, tokenizer, BLOCK_SIZE)
    tokenizer.model_max_length = model_max_length

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_amp,
    )

    config = GPTConfig(
        vocab_size=len(tokenizer),
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    model = QraXAiForCausalLM(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Cihaz: {device} | bf16: {use_amp}")
    print(f"Parametre: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Veri: {len(dataset)} parça | {len(loader)} adım/dönem")

    model.train()
    total_steps = len(loader) * EPOCHS
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        running_loss, running_steps = 0.0, 0

        # tqdm: canlı it/s ve tahmini kalan süre gösterir.
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="adım",
            dynamic_ncols=True,
            disable=None,  # tty yoksa (log dosyası vb.) çubuğu kapat
        )

        for step, (input_ids, labels) in enumerate(progress, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # bf16 karışık hassasiyet: hesaplama bf16, ağırlıklar/gradyanlar fp32.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(input_ids=input_ids, labels=labels).loss

            (loss / GRAD_ACCUM).backward()

            # Gradyanları biriktir, belirli aralıkla güncelle.
            last_step = step == len(loader)
            if step % GRAD_ACCUM == 0 or last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Çubukta anlık ve ortalama kaybı göster.
            running_loss += loss.item()
            running_steps += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                ort=f"{running_loss / running_steps:.4f}",
            )

            # Adım aralığındaki ortalama kaybı kalıcı olarak logla.
            if step % LOG_INTERVAL == 0 or last_step:
                done_steps = (epoch - 1) * len(loader) + step
                elapsed = time.time() - train_start
                eta = elapsed / done_steps * (total_steps - done_steps)
                tqdm.write(
                    f"Epoch {epoch}/{EPOCHS} | Step {step}/{len(loader)} "
                    f"| Loss {running_loss / running_steps:.4f} "
                    f"| Geçen {format_duration(elapsed)} "
                    f"| Kalan ~{format_duration(eta)}"
                )
                running_loss, running_steps = 0.0, 0

    print(f"Eğitim bitti | Toplam süre: {format_duration(time.time() - train_start)}")

    # AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
    # ile yüklenebilmesi için config'e auto_map ekle.
    config.auto_map = {
        "AutoConfig": "configuration_qraxai.GPTConfig",
        "AutoModelForCausalLM": "model.QraXAiForCausalLM",
    }

    output_dir = Path(OUTPUT_DIR)
    model.save_pretrained(output_dir)      # config.json + model.safetensors
    tokenizer.save_pretrained(output_dir)  # tokenizer dosyaları

    # Hub'a yüklendiğinde uzak kod olarak çalışması için kaynak dosyalar
    shutil.copy(Path(__file__).with_name("model.py"), output_dir / "model.py")
    shutil.copy(
        Path(__file__).with_name("configuration_qraxai.py"),
        output_dir / "configuration_qraxai.py",
    )

    print(f"Model hazır: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
