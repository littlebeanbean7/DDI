import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pytrec_eval
import argparse
import glob, os
import pandas as pd
import math
from reranking.metrics import ndkl # Lower is better https://pypi.org/project/reranking/


# parser = argparse.ArgumentParser()
# # eg python eval.py --pred output/t5-small_bestmodel_2025-10-29_21-30-00_predictions.json
# parser.add_argument('--pred', type=str, required=True, help='Path to saved prediction JSON')
# args = parser.parse_args()

# prediction_path = args.pred




def main(prediction_path, cut_k = 10):
    output_csv = prediction_path.replace('.json', '_eval_results.csv')
    
    print(f"\n========Evaluation on {prediction_path} using eval.py========")
    
    def dedup_preserve_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
        
    # Load predictions
    with open(prediction_path, 'r') as f:
        data = json.load(f)
    
    
    print(f"\n========{prediction_path}========")
    ids = [item["id"] for item in data]
    true_labels = [item["true_labels"] for item in data]
    
    # Deduplicate and truncate predictions to top-k
    predictions = [
        dedup_preserve_order(item["predictions"])[:cut_k]   # <- apply cutoff here
        for item in data
    ]
    

    print(f"Loaded {len(ids)} tasks, applying cutoff @ {cut_k}")
    print(f"Example truncated prediction: {predictions[0]}")


    
    """ If we don't deduplicate, there will be situation like:
    
    y_pred contains duplicates like:
    ["Alice", "Bob", "Alice", "Charlie"]
    
    run[qid] = {f'd{doc_id}': 1.0/(rank+1) for rank, doc_id in enumerate(y_pred)}
    the key 'dAlice' appears twice, and only the last one is kept (with the lowest score).
    So 'Alice'’s rank 1 score (1.0/1 = 1.0) is overwritten by her later duplicate (1.0/3 = 0.333).
    """
    
    # Prepare pytrec_eval input
    qrel = {}
    run = {}
    
    for i, (y_true, y_pred) in enumerate(zip(true_labels, predictions)):
        qid = f'q{i}'
        qrel[qid] = {f'd{label}': 1 for label in y_true}
    
        run[qid] = {f'd{doc_id}': 1.0 / (rank + 1) for rank, doc_id in enumerate(y_pred)}
    
    # Set of metrics to compute
    # metrics = [
    #     'P_10',
    #     'recall_10', 
    #     'recip_rank',
    #     'ndcg_cut_10',
    #     'map'
    # ]
    
    # ============================================================
    # ========== pytrec_eval Metrics (P@k, R@k, NDCG@k) ==========
    # ============================================================


    # limit y_pred to top-k
    y_pred_cut = y_pred[:cut_k]
    run[qid] = {f'd{doc_id}': 1.0 / (rank + 1) for rank, doc_id in enumerate(y_pred_cut)}

    # dynamically build metrics for k=1..10
    metrics = [f'P_{k}' for k in range(1, 11)] \
            + [f'recall_{k}' for k in range(1, 11)] \
            + [f'ndcg_cut_{k}' for k in range(1, 11)] \
            + ['recip_rank', 'map']
    
    print("Evaluating with pytrec_eval...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    results = evaluator.evaluate(run)
    
    # convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df_mean = df.mean().to_frame(name='mean')
    
    # neatly display grouped metrics
    print("\n===== Precision@k =====")
    print(df_mean.loc[[f'P_{k}' for k in range(1, 11)]].round(4).T)
    
    print("\n===== Recall@k =====")
    print(df_mean.loc[[f'recall_{k}' for k in range(1, 11)]].round(4).T)
    
    print("\n===== NDCG@k =====")
    print(df_mean.loc[[f'ndcg_cut_{k}' for k in range(1, 11)]].round(4).T)
    
    print("\n===== Global Metrics =====")
    print(df_mean.loc[['recip_rank', 'map']].round(4).T)
    
    # ============================================================
    # ================== Compute NDKL & F1 =======================
    # ============================================================
    # ---- NDKL ----
    # define target distribution: uniform across all contributors
    unique_contributors = set(sum(true_labels, []))
    dict_p = {c: 1 / len(unique_contributors) for c in unique_contributors}
    
    # compute NDKL per test example
    ndkl_scores = [ndkl(y_pred, dict_p) for y_pred in predictions]
    mean_ndkl = np.mean(ndkl_scores)
    print(f"\nMean NDKL: {mean_ndkl:.4f}")
    
    # ---- F1 Scores ----
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score
    
    mlb = MultiLabelBinarizer()
    mlb.fit(true_labels + predictions)  # union of label spaces
    
    true_binarized = mlb.transform(true_labels)
    pred_binarized = mlb.transform(predictions)
    
    # zero_division=0 That argument prevents scikit-learn from throwing warnings or NaNs when a label has no positive samples in either the prediction or the ground truth — which can happen easily in multi-label tasks
    f1_micro = f1_score(true_binarized, pred_binarized, average='micro', zero_division=0)
    
    
    print(f"\nF1 Scores:")
    print(f"Micro: {f1_micro:.4f}")
    
    # ============================================================
    # ============= Combine all results & Save ===================
    # ============================================================
    
    # add global metrics to dataframe
    df_mean.loc['mean_ndkl'] = mean_ndkl
    df_mean.loc['f1_micro'] = f1_micro
    
    
    # save as CSV
    df_mean = df_mean.reset_index().rename(columns = {"index": "matrix", "mean": "result"})
    df_mean["result"]= round(df_mean["result"], 4)
    df_mean.to_csv(output_csv, index = False)
    print(f"\nAll evaluation results saved to: {output_csv}")


# candidates = sorted(glob.glob(f"output/*_predictions.json"), key=os.path.getmtime)
# prediction_path = candidates[-1]  # latest

candidates = sorted(glob.glob(f"output/rerank_orig_model/*_predictions_rerank_alpha*.json"), key=os.path.getmtime)
#candidates = ["output/t5-small_bestmodel_2025-10-28_21-40-29_predictions.json"]
if __name__ == "__main__":
    for file in candidates:
        main(prediction_path = file, cut_k = 5
            )