from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def get_last_non_empty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def extract_row(record: Dict[str, Any]) -> Dict[str, str] | None:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    user_message = next((msg for msg in conversations if msg.get("role") == "user"), None)
    assistant_message = next((msg for msg in conversations if msg.get("role") == "assistant"), None)

    if not user_message or not assistant_message:
        return None

    input_text = get_last_non_empty_line(str(user_message.get("content", "")))
    output_text = str(assistant_message.get("content", "")).strip()

    if not input_text or not output_text:
        return None

    return {
        "id": str(record.get("id", "")),
        "image_path": str(record.get("image", "")),
        "input": input_text,
        "output": output_text,
    }


def load_json_file(json_path: Path) -> List[Dict[str, str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"File {json_path} does not contain a JSON list at root level.")

    rows: List[Dict[str, str]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        row = extract_row(record)
        if row is not None:
            rows.append(row)

    return rows


def collect_rows(input_dir: Path) -> pd.DataFrame:
    all_rows: List[Dict[str, str]] = []

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    for json_file in json_files:
        all_rows.extend(load_json_file(json_file))

    df = pd.DataFrame(all_rows)
    return df.drop_duplicates(subset=["id", "image_path", "input", "output"])


def export_tsv(input_dir: str, output_tsv: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = collect_rows(input_path)
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estrae id, image_path, input e output da una cartella di file JSON e salva un TSV senza duplicati."
    )
    parser.add_argument("input_dir", help="Cartella contenente i file JSON")
    parser.add_argument("output_tsv", help="Percorso del file TSV di output")
    args = parser.parse_args()

    export_tsv(args.input_dir, args.output_tsv)


if __name__ == "__main__":
    main()