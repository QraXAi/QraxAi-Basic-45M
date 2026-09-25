"""
train.py'nin ürettiği klasörü Hugging Face'e yükler.

Bir kez giriş yapman yeterli:
    hf auth login

Kullanım:
    python push.py
"""

from pathlib import Path

from huggingface_hub import HfApi, whoami
from huggingface_hub.errors import LocalTokenNotFoundError

REPO_ID = "coderian/QraXAi-Basic-45M"
MODEL_DIR = "qraxai"        
PRIVATE = False


def main():
    model_dir = Path(MODEL_DIR)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"'{MODEL_DIR}' bulunamadı. Önce train.py çalıştır.")

    try:
        whoami()
    except LocalTokenNotFoundError:
        raise SystemExit(
            "Hugging Face'e giriş yapılmamış. Bir kez 'hf auth login' çalıştır."
        )

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, private=PRIVATE, exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=MODEL_DIR,
        commit_message="QraXAi modeli eklendi",
    )
    print(f"Yüklendi: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
