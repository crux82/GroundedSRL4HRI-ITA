import json
import os
import argparse
from collections import defaultdict


def update_prompt(entry, new_prompt):
    content = entry.get('conversations', [{}])[0].get('content', '')
    lines = content.strip().rsplit('\n', 1)
    input_ita = lines[-1] if len(lines) > 1 else content
    entry['conversations'][0]['content'] = new_prompt + '\n<image>\n' + input_ita
    return entry


def load_json_entries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Merge due dataset JSON con la stessa struttura.'
    )
    parser.add_argument('--dataset-a', required=True,
                        help='Path della prima cartella dataset')
    parser.add_argument('--dataset-b', required=True,
                        help='Path della seconda cartella dataset')
    parser.add_argument('--output-dataset', required=True,
                        help='Path della cartella di output')
    parser.add_argument('--prompt', required=True,
                        help='Path del file TXT con il nuovo prompt')
    args = parser.parse_args()

    with open(args.prompt, 'r', encoding='utf-8') as f:
        new_prompt = f.read()

    files_a = {}
    for root, dirs, files in os.walk(args.dataset_a):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            rel_path = os.path.relpath(os.path.join(root, filename), args.dataset_a)
            files_a[rel_path] = os.path.join(root, filename)

    files_b = {}
    for root, dirs, files in os.walk(args.dataset_b):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            rel_path = os.path.relpath(os.path.join(root, filename), args.dataset_b)
            files_b[rel_path] = os.path.join(root, filename)

    os.makedirs(args.output_dataset, exist_ok=True)

    non_test_paths = sorted(set(files_a.keys()) | set(files_b.keys()))
    test_paths_b = [p for p in files_b.keys() if '_test' in p]
    non_test_paths = [p for p in non_test_paths if '_test' not in p]

    only_one_warnings = []
    total_entries = 0
    files_processed = 0
    test_files_processed = 0

    for rel_path in non_test_paths:
        in_a = rel_path in files_a
        in_b = rel_path in files_b
        output_path = os.path.join(args.output_dataset, rel_path)

        merged = []
        if in_a and in_b:
            merged.extend(load_json_entries(files_a[rel_path]))
            merged.extend(load_json_entries(files_b[rel_path]))
        elif in_a:
            merged.extend(load_json_entries(files_a[rel_path]))
            only_one_warnings.append((rel_path, 'A'))
        else:
            merged.extend(load_json_entries(files_b[rel_path]))
            only_one_warnings.append((rel_path, 'B'))

        merged = [update_prompt(entry, new_prompt) for entry in merged]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=4, ensure_ascii=False)

        total_entries += len(merged)
        files_processed += 1

    for rel_path in sorted(test_paths_b):
        output_path = os.path.join(args.output_dataset, rel_path)
        entries = load_json_entries(files_b[rel_path])
        entries = [update_prompt(entry, new_prompt) for entry in entries]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=4, ensure_ascii=False)

        total_entries += len(entries)
        files_processed += 1
        test_files_processed += 1

    print(f"\nDone: {files_processed} files processed, {total_entries} total entries")
    if only_one_warnings:
        print(f"Warning — files present in only one folder ({len(only_one_warnings)}):")
        for path, src in only_one_warnings:
            print(f"  - {path} (only in {src})")

    if test_files_processed:
        print(f"  Test files copied from the second dataset: {test_files_processed}")
    else:
        print("  WARNING: no test files found in the second dataset.")


if __name__ == '__main__':
    main()
