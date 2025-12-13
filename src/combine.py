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
        return "".join(p.get("text", "") for p in parts)
    except Exception:
        return ""


def combine_records(directory):
    groups = defaultdict(lambda: {"instruction": "", "input": "", "output": ""})
    invalid = set()

    for file in Path(directory).glob("*.jsonl"):
        for obj in load_jsonl(file):
            key = obj.get("key")
            if not key:
                continue

            base = key.rsplit("_", 1)[0]
            suffix = key.split("_")[-1]

            if "error" in obj:
                invalid.add(base)
                continue

            resp = obj.get("response")
            if resp is None:
                invalid.add(base)
                continue

            text = extract_text(resp)

            if suffix == "instruction":
                groups[base]["instruction"] = text
            elif suffix == "input":
                groups[base]["input"] = text
            elif suffix == "output":
                groups[base]["output"] = text

    for bad in invalid:
        groups.pop(bad, None)

    return groups


def write_combined_json(directory, out_path):
    combined = combine_records(directory)
    out_path = Path(out_path)
    out_path.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    write_combined_json((Path(__file__).parent.parent / "batches" / "results"), "combined.json")
