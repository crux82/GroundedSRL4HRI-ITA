import pandas as pd
from pathlib import Path


def extract_commands_from_tsv(tsv_file_path, output_tsv_path):
    """
    Reads a TSV with id and input columns and saves a TSV with unique commands.
    """
    df = pd.read_csv(tsv_file_path, sep='\t')

    if 'input' not in df.columns:
        raise ValueError(f"TSV file {tsv_file_path} does not contain an 'input' column.")

    if 'id' not in df.columns:
        df['id'] = ''

    commands_df = df[['id', 'input']].copy()
    commands_df['id'] = commands_df['id'].astype(str)
    commands_df['input'] = commands_df['input'].astype(str).str.strip()
    commands_df = commands_df[commands_df['input'] != '']

    df_unique = commands_df.drop_duplicates()
    df_unique.to_csv(output_tsv_path, sep='\t', index=False, encoding='utf-8')

    print(f"Commands extracted: {len(commands_df)}")
    print(f"Unique commands: {len(df_unique)}")
    print(f"Duplicates removed: {len(commands_df) - len(df_unique)}")
    print(f"File saved to: {output_tsv_path}")

    return df_unique


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_commands.py <path_tsv_input> [path_tsv_output]")
        print("\nExamples:")
        print("  python extract_commands.py dataset_top_1.tsv")
        print("  python extract_commands.py dataset_top_1.tsv output/commands.tsv")
        sys.exit(1)

    input_file = sys.argv[1]
    tsv_output = sys.argv[2] if len(sys.argv) > 2 else "commands.tsv"

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_file}")
        sys.exit(1)

    df = extract_commands_from_tsv(input_file, tsv_output)
    print("\nFirst extracted rows:")
    print(df.head())
