from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

with open("./dataset/data.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        if i >= 130_000:
            break

        text = row["text"].strip()

        if text:
            f.write(text + "\n")
