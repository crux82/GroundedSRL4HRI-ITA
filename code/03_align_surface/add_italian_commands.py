from pathlib import Path
import pandas as pd
import argparse


def load_corrections(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Corrections file not found: {excel_path}")

    df_cor = pd.read_excel(excel_path)
    required = {"id", "translation", "correction"}
    missing = required - set(df_cor.columns)
    if missing:
        raise ValueError(f"Corrections file missing columns: {missing}")

    df_cor['correction'] = df_cor['correction'].fillna(df_cor['translation'])
    return df_cor


def add_input_ita_column(df: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    if 'id' not in df.columns:
        raise ValueError("Input TSV missing 'id' column")

    df["input_ita"] = df["id"].map(corrections.set_index("id")["correction"])
    df = df[["id", "image_path", "input", "input_ita", "output"]]
    return df


def process_directory(input_dir: Path, output_dir: Path, corrections: pd.DataFrame) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(input_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {input_dir}")

    total_rows = 0
    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep='\t')
        df_out = add_input_ita_column(df, corrections)
        out_path = output_dir / tsv_path.name
        df_out.to_csv(out_path, sep="\t", index=False)
        print(f"  Processed: {tsv_path.name} ({len(df_out)} rows) → {out_path}")
        total_rows += len(df_out)

    print(f"Done: {len(tsv_files)} files processed, {total_rows} total rows")


def main():
    parser = argparse.ArgumentParser(
        description="Add an 'input_ita' column to TSV files by mapping IDs to corrections from an Excel file."
    )
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Directory containing the original TSV files")
    parser.add_argument("--corrections", "-c", required=True,
                        help="Excel file with columns 'id', 'translation', 'correction'")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for enriched TSV files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    corrections_path = Path(args.corrections)
    output_dir = Path(args.output_dir)

    print(f"Loading corrections from: {args.corrections}")
    corrections = load_corrections(corrections_path)
    print(f"  Loaded {len(corrections)} correction entries")

    print(f"Processing TSV files from: {args.input_dir}")
    process_directory(input_dir, output_dir, corrections)


if __name__ == "__main__":
    main()
