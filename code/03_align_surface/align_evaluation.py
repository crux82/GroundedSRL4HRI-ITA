from collections import Counter
import pandas as pd
import ast

file_path = './code/evaluation/surface_extracted.tsv'
ground_truth_col = 'ref_extract'
lm_output_col = 'hyp_extract'

def calculate_metrics(gt_list, lm_list):
    gt_counter = Counter(gt_list)
    lm_counter = Counter(lm_list)
    
    tp = sum((gt_counter & lm_counter).values())
    
    fp_counter = lm_counter - gt_counter
    print("False Positive: ", fp_counter)
    
    fn_counter = gt_counter - lm_counter
    print("False Negative: ", fn_counter)
    print("\n")
    
    return tp, fp_counter, fn_counter

df = pd.read_csv(file_path, sep='\t')
df = df[df[ground_truth_col].notna()]

total_tp = total_fp = total_fn = 0

for idx, row in df.iterrows():
    gt_list = ast.literal_eval(row[ground_truth_col])
    lm_list = ast.literal_eval(row[lm_output_col]) if pd.notna(row[lm_output_col]) else []
    
    tp, fp_counts, fn_counts = calculate_metrics(gt_list, lm_list)
    
    total_tp += tp
    total_fp += sum(fp_counts.values())
    total_fn += sum(fn_counts.values())

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}\n")
