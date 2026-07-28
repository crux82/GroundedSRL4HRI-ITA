from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import torch
import argparse
import ast
import re
from pathlib import Path


SYSTEM_PROMPT = """
You will be given a list of sentences in Italian. The sentences represent commands addressed to a robot in a home environment, given by a human. 
Your task is to translate the following commands from English to Italian.

The commands must be translated according to the following rules:
    1. Translate all the words in the English sentence into Italian, do NOT leave any out.
    2. The main verb must be translated so that it sounds like a direct command, typically using the IMPERATIVE form when appropriate.
    3. Use the meaning of the words appropriate for a HOME CONTEXT.
    4. The translation of the command must NOT mix with translations of other commands.
    5. Do NOT add commas, periods, question marks, or exclamation marks.
    6. KEEP accented letters in the Italian sentence.
    7. SEPARATE words with apostrophes from the following word (i.e. nell' armadio).

You will always receive:
    1. A list containing key-command pairs.
    2. Both the key and the command are strings.

The output must follow these rules:
    1. Your output must be ONLY the list of pairs, in the following format:
        - [("key","translation"), ("key","translation"), ...]
    2. Do NOT include any explanations or reasoning.
    3. Maintain the same order of keys between input and output.
"""


def translate(commands_list: str, model, tokenizer, device) -> str:
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"/no_think {commands_list}"}
    ]

    model_inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    THINK_END_ID = 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        index = 0

    return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def translate_dataframe(df: pd.DataFrame, model, tokenizer, device,
                        batch_size: int = 10) -> pd.DataFrame:
    df_new = df.copy()
    df_new["translations"] = None

    rows_to_process = df_new.index.tolist()
    for i in tqdm(range(0, len(rows_to_process), batch_size),
                  desc="Processing batches"):
        batch_idx = rows_to_process[i:i + batch_size]
        batch_df = df_new.loc[batch_idx]
        batch_texts = list(
            batch_df[['id', 'input']].astype(str).itertuples(index=False, name=None)
        )

        batch_translations = translate(str(batch_texts), model, tokenizer, device)
        batch_translations = re.sub(r"(\w)'\s*(\w)", r"\1' \2", batch_translations)

        try:
            translations_list = ast.literal_eval(batch_translations)

            for j, idx in enumerate(batch_idx):
                if j < len(translations_list):
                    df_new.at[idx, "translations"] = translations_list[j][-1]

        except Exception as e:
            print(f"Error at batch starting at index {i}: {e}")
            print(f"Raw output: {batch_translations}")

    return df_new


def main():
    parser = argparse.ArgumentParser(
        description="Translate commands using Qwen/Qwen3-8B"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input TSV with 'id' and 'input' columns")
    parser.add_argument("--output", "-o", required=True,
                        help="Output TSV path")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Batch size for translation (default: 10)")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    device = model.device

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input, sep='\t')

    print(f"Translating {len(df)} commands...")
    df_result = translate_dataframe(df, model, tokenizer, device, args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(args.output, sep='\t', index=False)
    print(f"Translations saved to: {args.output}")


if __name__ == "__main__":
    main()
