import json
import os
import re
import argparse


def extract_number(id_val):
    match = re.match(r'(\d+)', id_val)
    return match.group(1) if match else id_val


def add_suffix(id_val, suffix):
    if not suffix:
        return id_val
    dot_index = id_val.rfind('.')
    if dot_index != -1:
        return id_val[:dot_index] + suffix + id_val[dot_index:]
    return id_val + suffix


def load_tsv_lookup(tsv_path):
    lookup = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        try:
            id_idx = header.index('id')
            image_idx = header.index('image_path')
            input_eng_idx = header.index('input')
            output_eng_idx = header.index('output')
            input_ita_idx = header.index('input_ita')
            output_ita_idx = header.index('output_ita')
        except ValueError as e:
            raise ValueError(f"Missing column in TSV header: {e}")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            row_id = parts[id_idx].strip()
            row_image = parts[image_idx].strip()
            row_input_eng = parts[input_eng_idx].strip()
            row_output_eng = parts[output_eng_idx].strip()
            row_input_ita = parts[input_ita_idx].strip()
            row_output_ita = parts[output_ita_idx].strip()
            row_num = extract_number(row_id)
            key = (row_num, row_image)
            if key in lookup:
                raise ValueError(
                    f"Chiave duplicata nel TSV: numero='{row_num}', image='{row_image}'"
                )
            lookup[key] = (row_id, row_input_eng, row_output_eng, row_input_ita, row_output_ita)

    return lookup


def process_entry(entry, lookup, prompt, suffix):
    ref_id = entry.get('id', '')
    ref_image = entry.get('image', '')
    ref_num = extract_number(ref_id)

    key = (ref_num, ref_image)
    tsv_row = lookup.get(key)

    if tsv_row is None:
        return None

    tsv_id, _, _, input_ita, output_ita = tsv_row

    new_id = add_suffix(tsv_id, suffix) if suffix else tsv_id

    return {
        'id': new_id,
        'image': ref_image,
        'conversations': [
            {
                'role': 'user',
                'content': prompt + '\n<image>\n' + input_ita
            },
            {
                'role': 'assistant',
                'content': output_ita
            }
        ]
    }


def process_json_file(json_path, lookup, prompt, suffix):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    unmatched = []

    for entry in data:
        result = process_entry(entry, lookup, prompt, suffix)
        if result is not None:
            new_data.append(result)
        else:
            unmatched.append(entry.get('id', 'unknown'))

    return new_data, unmatched


def update_prompt_keep_original(entry, new_prompt, suffix):
    content = entry.get('conversations', [{}])[0].get('content', '')
    input_text = content.strip().rsplit('\n', 1)[-1]
    output_text = entry.get('conversations', [{}, {}])[1].get('content', '')
    ref_id = entry.get('id', '')
    ref_image = entry.get('image', '')
    new_id = add_suffix(ref_id, suffix) if suffix else ref_id
    return {
        'id': new_id,
        'image': ref_image,
        'conversations': [
            {'role': 'user', 'content': new_prompt + '\n<image>\n' + input_text},
            {'role': 'assistant', 'content': output_text}
        ]
    }


def process_test_files(test_entries, lookup, prompt, suffix, output_dataset, filename):
    seen = set()
    ita_data = []
    eng_data = []
    unmatched = []

    for entry in test_entries:
        ref_image = entry.get('image', '')
        ref_id = entry.get('id', '')
        ref_num = extract_number(ref_id)
        key = (ref_num, ref_image)

        if key in seen:
            continue
        seen.add(key)

        eng_entry = update_prompt_keep_original(entry, prompt, suffix)
        eng_data.append(eng_entry)

        tsv_row = lookup.get(key)
        if tsv_row is None:
            unmatched.append(ref_id)
            continue

        tsv_id, _, _, input_ita, output_ita = tsv_row
        new_id = add_suffix(tsv_id, suffix) if suffix else tsv_id

        ita_entry = {
            'id': new_id,
            'image': ref_image,
            'conversations': [
                {'role': 'user', 'content': prompt + '\n<image>\n' + input_ita},
                {'role': 'assistant', 'content': output_ita}
            ]
        }
        ita_data.append(ita_entry)

    ita_name = re.sub(r'_test\.json$', r'_test_ita.json', filename)
    eng_name = re.sub(r'_test\.json$', r'_test_eng.json', filename)

    os.makedirs(output_dataset, exist_ok=True)
    with open(os.path.join(output_dataset, ita_name), 'w', encoding='utf-8') as f:
        json.dump(ita_data, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dataset, eng_name), 'w', encoding='utf-8') as f:
        json.dump(eng_data, f, indent=4, ensure_ascii=False)

    return len(ita_data), len(eng_data), unmatched


def main():
    parser = argparse.ArgumentParser(
        description='Crea un dataset JSON identico nella struttura a uno di origine, '
                    'sostituendo i dati con quelli di un TSV.'
    )
    parser.add_argument('--tsv', required=True, help='Path del file TSV di input')
    parser.add_argument('--prompt', required=True, help='Path del file TXT con il prompt')
    parser.add_argument('--input-dataset', required=True,
                        help='Path della cartella del dataset di origine')
    parser.add_argument('--output-dataset', required=True,
                        help='Path della cartella del dataset di destinazione')
    parser.add_argument('--suffix', default='',
                        help='Suffisso negli ID del JSON di origine (es. _ita)')
    args = parser.parse_args()

    with open(args.prompt, 'r', encoding='utf-8') as f:
        prompt = f.read()

    print("Loading TSV into memory...")
    lookup = load_tsv_lookup(args.tsv)
    print(f"TSV loaded: {len(lookup)} rows")

    total_entries = 0
    total_unmatched = []
    files_processed = 0
    test_entries = []
    test_filename = None
    has_test_json = False
    has_lang_test_files = False

    for root, dirs, files in os.walk(args.input_dataset):
        for filename in files:
            if not filename.endswith('.json'):
                continue

            json_path = os.path.join(root, filename)
            rel_path = os.path.relpath(json_path, args.input_dataset)

            if re.search(r'_test\.json$', filename):
                has_test_json = True
                if test_filename is None:
                    test_filename = filename
                print(f"  Accumulating: {rel_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    test_entries.extend(json.load(f))
            elif re.search(r'_test_\w+\.json$', filename):
                has_lang_test_files = True
                print(f"  Skipping (unnamed): {rel_path}")
            else:
                output_path = os.path.join(args.output_dataset, rel_path)
                print(f"  Processing: {rel_path}")
                new_data, unmatched = process_json_file(json_path, lookup, prompt, args.suffix)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=4, ensure_ascii=False)

                total_entries += len(new_data)
                total_unmatched.extend(unmatched)
                files_processed += 1

    if has_lang_test_files and not has_test_json:
        print("\n  WARNING: found '*_test_ita.json' or '*_test_eng.json' files but no")
        print("           '*_test.json' file. Rename the test file to '*_test.json' and try again.")

    if test_entries:
        print(f"\n  Processing test file ({len(test_entries)} entries)...")
        ita_count, eng_count, test_unmatched = process_test_files(
            test_entries, lookup, prompt, args.suffix,
            args.output_dataset, test_filename
        )
        total_unmatched.extend(test_unmatched)
        print(f"    -> test_ita: {ita_count}, test_eng: {eng_count}")
        if test_unmatched:
            print(f"    -> {len(test_unmatched)} unmatched in test entries")

    print(f"\nDone: {files_processed} regular files processed")
    if test_entries:
        print(f"  Test: {ita_count} entry in test_ita, {eng_count} entry in test_eng")
    if total_entries:
        print(f"  Regular: {total_entries} entries created")
    if total_unmatched:
        print(f"IDs without TSV match ({len(total_unmatched)}):")
        for uid in sorted(set(total_unmatched)):
            count = total_unmatched.count(uid)
            print(f"  - {uid} ({count}x)")
    else:
        print("No unmatched IDs.")


if __name__ == '__main__':
    main()
