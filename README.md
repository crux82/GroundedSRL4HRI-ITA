# Grounded SRL for Human-Robot Interaction in Italian

_Adapting Grounded Semantic Role Labelling to Italian Situated Robot Commands_

A complete pipeline for adapting English datasets for **Grounded Semantic Role Labeling (G-SRL)** to a target language (Italian in this implementation). The repository provides the methods to extract commands, translate them using LLM and MT models, align semantic structures, and generate monolingual (target language) or multilingual (English + target language) datasets for training and evaluation.

---

## Overview

The pipeline supports:

- Data extraction from the source dataset (`code/01_extract_json_dataset_to_tsv.py`)
- Command extraction and translation using LLM and MT models (`code/02_command_extraction_and_translation/`)
- Alignment of structured semantic outputs to the target language (`code/03_align_surface/`)
- Generation of monolingual and multilingual datasets (`code/04_create_dataset/`)

---

## Installation (via Conda)

### Prerequisites

- CUDA-capable GPU
- NVIDIA CUDA drivers installed
- Conda (recommended) or pip

### Setup Commands

```bash
conda env create -f environment.yml
conda activate dataset_adaptation
./install_requirements.sh
```

Alternatively, install directly with pip:

```bash
pip install -r requirements.txt
```

---

## Step 1: Data Extraction

`code/01_extract_json_dataset_to_tsv.py`

First, download the dataset in this path `data\datasets\dataset_eng_peng`

Extracts `id`, `image_path`, `input` (command) and `output` (semantic structures) from JSON files into a TSV.

**Input:** Directory with the dataset in English in JSON format.

**Output:** TSV with columns `id`, `image_path`, `input`, `output`

```bash
python code/01_extract_json_dataset_to_tsv.py <input_dir> <output_tsv>
```

---

## Step 2: Command Extraction & Translation

### 2a. Extract Commands

`code/02_command_extraction_and_translation/extract_commands.py`

Extracts unique `(id, input)` pairs from the Phase 1 TSV, removing empty and duplicate entries.

**Input:** TSV with columns `id`, `input`

**Output:** TSV file containing the list of unique commands.

```bash
python code/02_command_extraction_and_translation/extract_commands.py <input_tsv> [output_tsv]
```

### 2b. Translate Commands

Commands are translated from English to Italian using the following models:

| Script / Notebook                        | Model                              |
| ---------------------------------------- | ---------------------------------- |
| `translators_cli/translate_madlad400.py` | `google/madlad400-7b-mt`           |
| `translators_cli/translate_nllb200.py`   | `facebook/nllb-200-3.3B`           |
| `translators_cli/translate_llama.py`     | `meta-llama/Llama-3.1-8B-Instruct` |
| `translators_cli/translate_qwen2_5.py`   | `Qwen/Qwen2.5-7B-Instruct`         |
| `translators_cli/translate_qwen3.py`     | `Qwen/Qwen3-8B`                    |

Notebook equivalents are in `translators/` for GPU environments (Kaggle/Colab). CLI scripts support headless execution:

```bash
python code/02_command_extraction_and_translation/translators_cli/translate_madlad400.py \
    --input data/tsv/commands.tsv --output data/translations/madlad400.tsv

python code/02_command_extraction_and_translation/translators_cli/translate_nllb200.py \
    --input data/tsv/commands.tsv --output data/translations/nllb200.tsv

python code/02_command_extraction_and_translation/translators_cli/translate_llama.py \
    --input data/tsv/commands.tsv --output data/translations/llama.tsv --hf-token <token>

python code/02_command_extraction_and_translation/translators_cli/translate_qwen2_5.py \
    --input data/tsv/commands.tsv --output data/translations/qwen2_5.tsv

python code/02_command_extraction_and_translation/translators_cli/translate_qwen3.py \
    --input data/tsv/commands.tsv --output data/translations/qwen3.tsv
```

### 2c. Evaluate Translations

**BLEU Score** — `code/02_command_extraction_and_translation/evaluation/bleu_score.py`

Configure `TRANSLATIONS_DIR` and `GOLD_REFERENCES_FILE` in the script, then run:

```bash
python code/02_command_extraction_and_translation/evaluation/bleu_score.py
```

**Sentence Similarity** — `code/02_command_extraction_and_translation/evaluation/sentence-similarity-score.ipynb`

Configure `TRANSLATIONS_DIR` and `REFERENCES_FILE` in the notebook. Computes average cosine similarity via SentenceTransformer.

---

## Step 3: Surface Alignment

### 3a. Add Italian Commands

`code/03_align_surface/add_italian_commands.py`

Adds an `input_ita` column to the Phase 1 TSVs by mapping IDs to corrections from a reference Excel file.

**Output:** TSV with columns `id`, `image_path`, `input`, `input_ita`, `output`

```bash
python code/03_align_surface/add_italian_commands.py \
    --input-dir data/tsv/extracted \
    --corrections data/corrections/file.xlsx \
    --output-dir data/tsv/input_ita
```

| Argument               | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `--input-dir` / `-i`   | Directory with original TSV files                                                             |
| `--corrections` / `-c` | Excel file containing the correct commands with the columns `id`, `translation`, `correction` |
| `--output-dir` / `-o`  | Output directory for enriched TSV files                                                       |

### 3b. Align Surfaces

`code/03_align_surface/align_surface_cli.py` (notebook: `align-surface-qwen.ipynb`)

Uses Qwen3-8B to map each English surface in the structured JSON frames to its corresponding Italian word, producing the `output_ita` column.

**Input:** TSV with columns `id`, `image_path`, `input`, `input_ita`, `output`

**Output:** TSV with additional column `output_ita`

```bash
python code/03_align_surface/align_surface_cli.py \
    --input data/tsv/input_ita.tsv \
    --output data/tsv/aligned.tsv
```

### 3c. Evaluate Alignment

`code/03_align_surface/align_evaluation.py`

Computes Precision, Recall, and F1 for surface extraction compared to ground truth.

**Setup:** Configure `file_path`, `ground_truth_col`, `lm_output_col` in the script.

```bash
python code/03_align_surface/align_evaluation.py
```

---

## Step 4: Dataset Creation

### 4a. Monolingual (Target Language)

`code/04_create_dataset/tsv_to_json_dataset.py`

Converts the enriched TSV into JSON format, structurally identical to the original dataset. Test files produce both `*_test_ita.json` and `*_test_eng.json`.

```bash
python code/04_create_dataset/tsv_to_json_dataset.py \
    --tsv data/tsv/aligned.tsv \
    --prompt code/04_create_dataset/prompt/prompt_sample.txt \
    --input-dataset data/datasets/english_original \
    --output-dataset data/datasets/italian \
    --suffix _ita
```

| Argument           | Description                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `--tsv`            | Enriched TSV form Phase 3 (`id`, `image_path`, `input`, `output`, `input_ita`, `output_ita`) |
| `--prompt`         | TXT prompt template file                                                                     |
| `--input-dataset`  | Original JSON dataset directory                                                              |
| `--output-dataset` | Output directory                                                                             |
| `--suffix`         | ID suffix (e.g., `_ita`)                                                                     |

### 4b. Multilingual (English + Target)

`code/04_create_dataset/merge_datasets.py`

Merges two JSON datasets with the same directory structure, applying a common prompt.

```bash
python code/04_create_dataset/merge_datasets.py \
    --dataset-a data/datasets/english_original \
    --dataset-b data/datasets/italian \
    --output-dataset data/datasets/multilingual \
    --prompt code/04_create_dataset/prompt/prompt_sample.txt
```

| Argument           | Description                    |
| ------------------ | ------------------------------ |
| `--dataset-a`      | First dataset (e.g., English)  |
| `--dataset-b`      | Second dataset (e.g., Italian) |
| `--output-dataset` | Output directory               |
| `--prompt`         | TXT prompt template file       |

---

## Model Training

The datasets produced by this pipeline are designed for fine-tuning multimodal models on **Grounded Semantic Role Labeling (G-SRL)**. The training code and methodology are available in the [GroundedSRL4HRI](https://github.com/crux82/GroundedSRL4HRI) repository, refer to the [`training_models/README.md`](https://github.com/crux82/GroundedSRL4HRI/blob/master/training_models/README.md) for full configuration and usage details.

---

## Project Structure

```
code/
├── 01_extract_json_dataset_to_tsv.py
├── 02_command_extraction_and_translation/
│   ├── extract_commands.py
│   ├── evaluation/
│   │   ├── bleu_score.py
│   │   └── sentence-similarity-score.ipynb
│   ├── translators/
│   └── translators_cli/
├── 03_align_surface/
│   ├── add_italian_commands.py
│   ├── align-surface-qwen.ipynb
│   ├── align_surface_cli.py
│   └── align_evaluation.py
└── 04_create_dataset/
    ├── tsv_to_json_dataset.py
    ├── merge_datasets.py
    └── prompt/
        ├── english_prompt.txt
        └── italian_prompt.txt
```

---

## Citation

```bibtex

```

---

## References

_Claudiu Daniel Hromei, Antonio Scaiella, Danilo Croce, Roberto Basili_ **Grounded Semantic Role Labelling from Synthetic Multimodal Data for Situated Robot Commands**. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 23758 - 23781, Suzhou, China, 2025
