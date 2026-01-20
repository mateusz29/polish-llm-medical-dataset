import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    for line in path.open("r", encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line:
            yield json.loads(line)


def extract_text(response):
    try:
        parts = response["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text.strip()
    except Exception:
        return ""


def extract_source_text(obj):
    try:
        raw_text = obj["request"]["contents"][0]["parts"][0]["text"]
        prefix = "Text to translate:\n"
        if raw_text.startswith(prefix):
            raw_text = raw_text[len(prefix):]
        return raw_text.strip()
    except Exception:
        return ""


def combine_records(results_dir, source_dir):
    groups = defaultdict(lambda: {
        "instruction": {"translated": "", "original": ""},
        "input": {"translated": "", "original": ""},
        "output": {"translated": "", "original": ""}
    })
    invalid_ids = set()

    for file in Path(source_dir).glob("*.jsonl"):
        for obj in load_jsonl(file):
            key = obj.get("key")
            if not key:
                continue
            base = key.rsplit("_", 1)[0]
            suffix = key.split("_")[-1]
            source_text = extract_source_text(obj)
            if suffix in ["instruction", "input", "output"]:
                groups[base][suffix]["original"] = source_text

    for file in Path(results_dir).glob("*.jsonl"):
        for obj in load_jsonl(file):
            key = obj.get("key")
            if not key:
                continue

            base = key.rsplit("_", 1)[0]
            suffix = key.split("_")[-1]

            if "error" in obj or obj.get("response") is None:
                invalid_ids.add(base)
                continue

            translated_text = extract_text(obj["response"])
            if suffix in ["instruction", "input", "output"]:
                groups[base][suffix]["translated"] = translated_text

    final_data = {}
    for base, content in groups.items():
        if base in invalid_ids:
            continue
        
        is_complete = True
        for section in ["instruction", "input", "output"]:
            if not content[section]["translated"] or not content[section]["original"]:
                is_complete = False
                break
        
        if is_complete:
            final_data[base] = content

    print(invalid_ids)
    return final_data


def write_combined_json(results_dir, source_dir, out_path):
    combined = combine_records(results_dir, source_dir)
    out_path = Path(out_path)
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    results_path = Path(__file__).parent.parent / "batches" / "old" / "results"
    source_path = Path(__file__).parent.parent / "batches" / "old" / "done"
    write_combined_json(results_path, source_path, "originals_and_translations.json")
