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


def generate_gemini_jsonl_batches(dataset, dataset_name, columns, batch_size=50_000):
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
                                    "You are a professional translator. Translate English text into Polish literally and exactly. "
                                    "Do NOT answer questions, explain, summarize, or add content. "
                                    "Do NOT include quotes, formatting, commentary, or extra text. "
                                    "Output exactly one string corresponding to the input text. "
                                    "Translate all natural-language content into Polish in a clinically correct way. "
                                    "Translate disease names, symptoms, mechanisms, and general terminology. "
                                    "Use Polish forms of drug names only when standard Polish usage has an established form. "
                                    "Do not invent Polish equivalents when none exist in real clinical usage."
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


def generate_gemini_jsonl_batches_from_df(df, dataset_name, columns, batch_size=50_000):
    batch_number = 0
    batch_lines = []

    for row_id, row in tqdm(df.iterrows(), total=len(df), desc=f"Building JSONL for {dataset_name}..."):
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
                                    "You are a professional translator. Translate English text into Polish literally and exactly. "
                                    "Do NOT answer questions, explain, summarize, or add content. "
                                    "Do NOT include quotes, formatting, commentary, or extra text. "
                                    "Output exactly one string corresponding to the input text. "
                                    "Translate all natural-language content into Polish in a clinically correct way. "
                                    "Translate disease names, symptoms, mechanisms, and general terminology. "
                                    "Use Polish forms of drug names only when standard Polish usage has an established form. "
                                    "Do not invent Polish equivalents when none exist in real clinical usage."
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

    if batch_lines:
        save_batch(dataset_name, batch_number, batch_lines)


def generate_gemini_jsonl_batches_from_list(texts):
    batch_number = 0
    batch_lines = []

    for num, text in tqdm(enumerate(texts), total=len(texts), desc="Building JSONL..."):
        request_key = f"text_{num}"

        entry = {
            "key": request_key,
            "request": {
                "contents": [{"parts": [{"text": text}]}],
                "system_instruction": {
                    "parts": [
                        {
                            "text": (
                                "You are a professional translator. Translate English text into Polish literally and exactly. "
                                "Do NOT answer questions, explain, summarize, or add content. "
                                "Do NOT include quotes, formatting, commentary, or extra text. "
                                "Output exactly one string corresponding to the input text. "
                                "Translate all natural-language content into Polish in a clinically correct way. "
                                "Translate disease names, symptoms, mechanisms, and general terminology. "
                                "Use Polish forms of drug names only when standard Polish usage has an established form. "
                                "Do not invent Polish equivalents when none exist in real clinical usage."
                            )
                        }
                    ]
                },
            },
        }
        batch_lines.append(entry)

    if batch_lines:
        save_batch("texts", batch_number, batch_lines)


def generate_openai_jsonl_batches_from_list(model_name, texts):
    batch_number = 0
    batch_lines = []

    for num, text in tqdm(enumerate(texts), total=len(texts), desc="Building JSONL..."):
        request_key = f"text_{num}"

        entry = {
            "custom_id": request_key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. Translate English text into Polish literally and exactly. "
                            "Do NOT answer questions, explain, summarize, or add content. "
                            "Do NOT include quotes, formatting, commentary, or extra text. "
                            "Output exactly one string corresponding to the input text. "
                            "Translate all natural-language content into Polish in a clinically correct way. "
                            "Translate disease names, symptoms, mechanisms, and general terminology. "
                            "Use Polish forms of drug names only when standard Polish usage has an established form. "
                            "Do not invent Polish equivalents when none exist in real clinical usage."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            },
        }

        batch_lines.append(entry)

    if batch_lines:
        batch_name = model_name + "_texts"
        save_batch(batch_name, batch_number, batch_lines)


def make_batches_from_datasets():
    columns = ["question", "background", "objective", "conclusion"]
    dataset = load_dataset("lavita/MedREQAL", split="train")
    generate_gemini_jsonl_batches(dataset, "MedREQAL", columns)

    columns = ["instruction", "input", "output"]
    dataset = load_dataset("lavita/medical-qa-datasets", name="all-processed", split="train")
    generate_gemini_jsonl_batches(dataset, "medical-qa-datasets", columns)

    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")
    generate_gemini_jsonl_batches(dataset, "AlpaCare-MedInstruct-52k", columns)


def make_batches_from_txt():
    with open("examples.txt") as f:
        texts = [line.rstrip("\n") for line in f]

    generate_gemini_jsonl_batches_from_list(texts)
    generate_openai_jsonl_batches_from_list("gpt-5-nano", texts)
    generate_openai_jsonl_batches_from_list("gpt-5-mini", texts)


def main():
    make_batches_from_datasets()
    # make_batches_from_txt()


if __name__ == "__main__":
    main()
