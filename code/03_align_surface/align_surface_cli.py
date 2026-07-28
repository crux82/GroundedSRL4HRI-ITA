from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict, deque
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import torch
import argparse
import json
import ast
import re
from pathlib import Path


SYSTEM_PROMPT = """
TASK DESCRIPTION
Extract lexical mappings from a structured JSON by aligning each extracted English surface form to its best-matching SINGLE token in the provided Italian sentence. 
Produce pairs for ordinary tokens and triples for third-person pronouns, preserving the exact traversal order of the JSON.

INPUT
You are given three items:
1) English sentence: <EN_TEXT>
2) Italian sentence: <IT_TEXT>
3) ArrayJSON: <ARRAY_JSON>
- ArrayJSON is a list of frames; each frame has an "elements" list; each element has a "surface" field.
- Only the "surface" field is eligible for extraction; ignore all other fields (e.g. frame, name).

RULES
- Extraction:
  - Traverse the ArrayJSON in order; within each frame, process elements in their listed order.
  - For each element, extract the exact string in its "surface" field; call it surface_extracted.
- Mapping to Italian:
  - IMPORTANT: Extract ONLY ONE token from <IT_TEXT> (e.g., "sala" for "living room" and NOT "sala da pranzo").
  - For ordinary tokens, return the SINGLE Italian token from <IT_TEXT> that best matches surface_extracted in meaning and role; (e.g., "sacchetti" for "bags", "cassetto" for "drawer").
  - Operational state keywords: if surface_extracted is "on" or "off" referring to device state, return it verbatim as the Italian token ("<ON>" or "<OFF>"), regardless of its presence in <IT_TEXT>.
  - Missing matches: if no suitable Italian token for surface_extracted is present in <IT_TEXT>, set italian_word to '' (empty string). Do NOT invent or infer elided/implied tokens.
  - Pronouns: if surface_extracted is a third-person pronoun (e.g., "it", "they", "her", "this"...):
    - italian_word = the Italian pronoun token exactly as it appears in <IT_TEXT>; if it is a clitic attached to a verb (e.g., "portarlo", "cercarla", "prendili"), extract only the clitic itself ("lo", "la", "li" etc.). If no pronoun token is present in <IT_TEXT>, set italian_word to ''.
    - reference = the SINGLE Italian token in <IT_TEXT> that is the antecedent of the pronoun, chosen by nearest suitable context alignment with <EN_TEXT>. If no antecedent token is present in <IT_TEXT>, set reference to ''.
- Do not introduce any new surfaces: only process surfaces that appear in the JSON. Keep duplicates if they recur.
- Maintain order strictly identical to the JSON traversal order.
- No commentary, no reasoning, no extra tokens.

OUTPUT
- Produce a single list:
  - For ordinary tokens and operational state keywords: output a pair (surface_extracted, italian_word).
  - For third-person pronouns: output a triple (surface_extracted, italian_word, reference).
- Use parentheses and single quotes exactly, comma-separated, in a single bracketed list.
- Output only the list, nothing else.
"""


def normalize_output(output_str: str) -> str:
    try:
        data = json.loads(output_str.replace("'", '"'))
    except Exception:
        return output_str

    def remove_bbox(obj):
        if isinstance(obj, dict):
            return {k: remove_bbox(v) for k, v in obj.items() if k != "bbox_2d"}
        elif isinstance(obj, list):
            return [remove_bbox(x) for x in obj]
        return obj

    clean_data = remove_bbox(deepcopy(data))
    return repr(clean_data)


def group_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["output_clean"] = df["output"].apply(normalize_output)

    grouped = (
        df.groupby(["input", "input_ita", "output_clean"],
                    sort=False, group_keys=False)
        .agg({
            "id": lambda x: list(x),
            "image_path": lambda x: list(x),
            "output": lambda x: list(x)
        })
        .reset_index()
    )
    return grouped


def align_frame(en_text: str, ita_text: str, frame: str,
                model, tokenizer, device) -> str:
    user_content = f"English text: {en_text}\nItalian text: {ita_text}\nArrayJSON: {frame}"

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"/no_think {user_content}"}
    ]

    model_inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=124,
            eos_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    THINK_END_ID = 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        index = 0

    return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def replace_surfaces(mapping_str: str, frames_str: str) -> str:
    mapping = ast.literal_eval(mapping_str)
    frames = ast.literal_eval(frames_str)

    mapping_dict = defaultdict(deque)
    for m in mapping:
        if len(m) == 2:
            orig, trans = m
            if trans == "":
                mapping_dict[orig].append({"remove": True})
            else:
                mapping_dict[orig].append({"surface": trans})
        elif len(m) == 3:
            orig, trans, ref = m
            if trans == "":
                mapping_dict[orig].append({"remove": True})
            else:
                mapping_dict[orig].append({"surface": trans, "reference": ref})

    for frame in frames:
        new_elements = []
        for el in frame["elements"]:
            surf = el["surface"]
            if mapping_dict[surf]:
                replacement = mapping_dict[surf].popleft()
                if "remove" in replacement:
                    continue
                el.update(replacement)
            new_elements.append(el)
        frame["elements"] = new_elements

    return str(frames)


def process_file(df: pd.DataFrame, model, tokenizer, device,
                 output_file: str, save_interval: int = 50) -> pd.DataFrame:
    df_new = df.copy()
    df_new["output_ita"] = None

    grouped_df = group_dataframe(df)

    counter = 0
    for _, row in tqdm(grouped_df.iterrows(), total=len(grouped_df),
                        desc="Processing"):
        counter += 1

        input_str = str(row["input"])
        input_ita_str = str(row["input_ita"])
        output_cleaned = str(row["output_clean"])

        try:
            result = align_frame(input_str, input_ita_str, output_cleaned,
                                 model, tokenizer, device)

            for id_val, img_val in zip(row["id"], row["image_path"]):
                mask = (df_new["id"] == id_val) & (df_new["image_path"] == img_val)
                if mask.any():
                    output_str = df_new.loc[mask, "output"].iloc[0]
                    output_ita = replace_surfaces(result, output_str)
                    df_new.loc[mask, "output_ita"] = str(output_ita)

        except Exception as e:
            print(f"Error: {e}")
            print(f"Row ID: {row['id'][0]}")
        finally:
            if counter >= save_interval:
                df_new.to_csv(output_file, sep="\t", index=False)
                print(f"Partial save at row {row['id'][0]}")
                counter = 0

    df_new.to_csv(output_file, sep='\t', index=False)
    print(f"Final file saved to: {output_file}")

    return df_new


def main():
    parser = argparse.ArgumentParser(
        description="Align surface forms in structured JSON outputs to Italian"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input TSV with columns: id, image_path, input, input_ita, output")
    parser.add_argument("--output", "-o", required=True,
                        help="Output TSV path (with output_ita column added)")
    parser.add_argument("--batch-size", "-b", type=int, default=50,
                        help="Save interval (number of groups, default: 50)")
    parser.add_argument("--model-name",
                        default="Qwen/Qwen3-8B",
                        help="Model name or path (default: Qwen/Qwen3-8B)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use (default: auto)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    device = model.device

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input, sep='\t')

    required_cols = {"id", "image_path", "input", "input_ita", "output"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input TSV missing columns: {missing}")

    print(f"Processing {len(df)} rows...")
    process_file(df, model, tokenizer, device, args.output, args.batch_size)


if __name__ == "__main__":
    main()
