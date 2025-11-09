import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def save_batch(dataset_name, batch_number, batch_lines):
    directory = Path(__file__).parent.parent / "batches"
    directory.mkdir(parents=True, exist_ok=True)

    filename = directory / f"{dataset_name}_batch_{batch_number}.jsonl"

    with open(filename, "w", encoding="utf-8") as f:
        for line in batch_lines:
            f.write(json.dumps(line) + "\n")


def generate_jsonl_batches(dataset, dataset_name, columns, batch_size=50_000):
    batch_number = 0
    batch_lines = []

    for row_id, row in tqdm(enumerate(dataset), total=len(dataset), desc=f"Building JSONL for {dataset_name} dataset..."):
        for col in columns:
            text = row[col]
            request_key = f"{dataset_name}_{row_id}_{col}"

            entry = {
                "key": request_key,
                "request": {
                    "contents": [{"parts": [{"text": text}]}],
                    "system_instruction": {
                        "parts": [
                            {
                                "text": (
                                    "You are a professional English-to-Polish translator. "
                                    "Translate all user messages from English to Polish. "
                                    "Only output the translated text, no explanations, formatting, or quotes."
                                )
                            }
                        ]
                    },
                },
            }
            batch_lines.append(entry)

        if len(batch_lines) >= batch_size:
            save_batch(dataset_name, batch_number, batch_lines)
            batch_number += 1
            batch_lines = []

    # save any remaining lines
    if batch_lines:
        save_batch(dataset_name, batch_number, batch_lines)


def main():
    columns = ["question", "background", "objective", "conclusion"]
    dataset = load_dataset("lavita/MedREQAL", split="train")
    generate_jsonl_batches(dataset, "MedREQAL", columns)

    columns = ["instruction", "input", "output"]
    dataset = load_dataset("lavita/medical-qa-datasets", name="all-processed", split="train")
    generate_jsonl_batches(dataset, "medical-qa-datasets", columns)

    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")
    generate_jsonl_batches(dataset, "AlpaCare-MedInstruct-52k", columns)


if __name__ == "__main__":
    main()
