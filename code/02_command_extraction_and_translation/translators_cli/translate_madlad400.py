from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import pandas as pd
import torch
import argparse
import re
from pathlib import Path


def translate(texts: list[str], model, tokenizer, device,
              lang_prefix: str = "<2it>") -> list[str]:
    input_texts = [f"{lang_prefix} Robot, {t}" for t in texts]

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            length_penalty=1.0,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    translations = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith(lang_prefix):
            text = text[len(lang_prefix):].strip()
        try:
            text = text.removeprefix("Robot, ").lstrip()
            text = re.sub(r"[^\w\s']", '', text, flags=re.UNICODE).lower()
            text = re.sub(r"\se'\s", r" è ", text)
            text = re.sub(r"(\w)'\s*(\w)", r"\1' \2", text).strip()
        except Exception:
            text = "TRANSLATION ERROR"
        translations.append(text)

    return translations


def translate_dataframe(df: pd.DataFrame, model, tokenizer, device,
                        batch_size: int = 10) -> pd.DataFrame:
    results = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i + batch_size]
        batch_texts = batch_df['input'].astype(str).tolist()
        batch_ids = batch_df['id'].tolist()

        batch_translations = translate(batch_texts, model, tokenizer, device)

        results.extend(zip(batch_ids, batch_texts, batch_translations))

    return pd.DataFrame(results, columns=["id", "input", "translation"])


def main():
    parser = argparse.ArgumentParser(
        description="Translate commands using google/madlad400-7b-mt"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input TSV with 'id' and 'input' columns")
    parser.add_argument("--output", "-o", required=True,
                        help="Output TSV path")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Batch size for translation (default: 10)")
    parser.add_argument("--model-name",
                        default="google/madlad400-7b-mt",
                        help="Model name or path (default: google/madlad400-7b-mt)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use (default: auto)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading model: {args.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
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
