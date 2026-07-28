import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
import numpy as np
#nltk.download('punkt')
#nltk.download("punkt_tab")

# Configure data paths
TRANSLATIONS_DIR = "path/to/translations"  # Directory containing translation .tsv files
GOLD_REFERENCES_FILE = "path/to/gold_reference.xlsx"  # Excel file with gold standard translations

def preprocess_text(text):
    if pd.isna(text) or text is None:
        return []
    
    text = str(text).lower().strip()
    
    return word_tokenize(text)

def calculate_bleu_score(df, smoothing_function=None, max_n=4):
    if smoothing_function is None:
        smoothing_function = SmoothingFunction().method1
    
    references, hypotheses, individual_scores = [], [], []
    
    for _, row in df.iterrows():
        ref_tokens = preprocess_text(row['translation_reference'])
        hyp_tokens = preprocess_text(row['translation_candidate'])
        
        if len(ref_tokens) > 0 and len(hyp_tokens) > 0:
            references.append([ref_tokens])
            hypotheses.append(hyp_tokens)
            weights = [(1.0,), (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]

            try:
                score = sentence_bleu(
					references=[ref_tokens],
					hypothesis=hyp_tokens,
					weights=weights,
					smoothing_function=smoothing_function
				)
                individual_scores.append((row['id'], score))
            except Exception as e:
                print(e)
                individual_scores.append((row['id'], [0.0, 0.0, 0.0, 0.0]))
    
    if individual_scores:
        scores_array = np.array([s for _, s in individual_scores])
        avg_sentence_score = scores_array.mean(axis=0).round(4).tolist()
    else:
        avg_sentence_score = [0.0, 0.0, 0.0, 0.0]
    
    return {
        'average_sentence_bleu': avg_sentence_score,
        'individual_scores': individual_scores,
        'valid_sentences': len(references),
        'total_sentences': len(df)
    }

def get_working_df(df_references, df_candidate):
    df_final = pd.merge(
        df_references,
        df_candidate,
        on=['id', 'input'],
        how='inner'
    )

    return df_final[df_final['translation_reference'].notna()]

def main():
    df_references = pd.read_excel(GOLD_REFERENCES_FILE)
    df_references = df_references.rename(columns={'translation': 'translation_reference'})

    for file in list(TRANSLATIONS_DIR.glob("*.tsv")):
        df_candidate = pd.read_csv(file, sep="\t")
        df_candidate = df_candidate.rename(columns={'translation': 'translation_candidate'})

        df_evaluation = get_working_df(df_references, df_candidate)
        
        results = calculate_bleu_score(df_evaluation)
        print(50*"-" + "\n" + f"Translation results for {file.name}: ")
        print(f"- Valid sentences: {results['valid_sentences']}/{results['total_sentences']}")
        print(f"- Average sentence BLEU: {results['average_sentence_bleu']}")

if __name__ == "__main__":
    main()