"""
Benchmark Standard BM25 Retrieval using rank_bm25.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Iterable
from tqdm import tqdm
import logging
from rank_bm25 import BM25Okapi

# =============== Config ===============
# Inputs
REFS_PARQUET = "./data/refs.parquet"
JSON_IN = "./data/sr4cs.json"

# Outputs
OUTPUT_SCORES = "./data/results/bm25_standard_scores.parquet"
LOG_FILE = "./logs/bm25_standard.log"

# BM25 Parameters (Standard Defaults)
K1 = 1.5
B = 0.75

# Retrieval Depth
RETRIEVAL_LIMIT = 1000
K_LEVELS = [1, 5, 10, 20, 50, 100, 1000]

# Preprocessing
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", 
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

# =============== Logging ===============
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
log = logging.getLogger("bm25_py")


# =============== Text Processing ===============
def simple_tokenize(text: str) -> List[str]:
    """
    Standard alphanumeric tokenization + lowercase + stopword removal.
    """
    if not isinstance(text, str):
        return []
    
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# =============== Metrics ===============
def _to_str_set(ids: Iterable) -> Set[str]:
    return {str(x).strip() for x in ids if str(x).strip()}

def evaluate(retrieved_ids_ordered: List[str], truth_ids: Iterable) -> dict:
    R_set = _to_str_set(retrieved_ids_ordered)
    T_set = _to_str_set(truth_ids)
    
    # --- Set Metrics ---
    tp = len(R_set & T_set)
    fp = len(R_set - T_set)
    fn = len(T_set - R_set)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(p, r, beta):
        if p == 0 and r == 0: return 0.0
        b2 = beta ** 2
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f3": f_beta(precision, recall, 3.0),
        "retrieved_count": len(R_set),
        "truth_count": len(T_set),
    }

    # --- Ranked Metrics ---
    hits_vector = [1 if doc_id in T_set else 0 for doc_id in retrieved_ids_ordered]
    num_relevant = len(T_set)
    
    # MAP
    cumulative_hits = 0
    sum_precisions = 0.0
    for i, is_hit in enumerate(hits_vector):
        if is_hit:
            cumulative_hits += 1
            sum_precisions += cumulative_hits / (i + 1)
            
    metrics["map"] = (sum_precisions / num_relevant) if num_relevant > 0 else 0.0

    # @k
    for k in K_LEVELS:
        hits_at_k = hits_vector[:k]
        tp_at_k = sum(hits_at_k)
        metrics[f"p@{k}"] = tp_at_k / k
        metrics[f"r@{k}"] = (tp_at_k / num_relevant) if num_relevant > 0 else 0.0

    return metrics


# =============== Main ===============
if __name__ == "__main__":
    log.info("=== Standard BM25 (Python) Run Started ===")
    
    # 1. Load Corpus (Parquet)
    log.info(f"Loading corpus from {REFS_PARQUET}...")
    df_refs = pd.read_parquet(REFS_PARQUET, columns=["ref_id", "title_norm", "abstract"])
    
    # --- CRITICAL FIX START ---
    # 1. Force numeric, coercing bad data to NaN
    df_refs["ref_id"] = pd.to_numeric(df_refs["ref_id"], errors='coerce')
    # 2. Drop rows that have no valid ID
    df_refs = df_refs.dropna(subset=["ref_id"])
    # 3. Convert to int (removes .0) then to string
    df_refs["ref_id"] = df_refs["ref_id"].astype(int).astype(str)
    # --- CRITICAL FIX END ---
    
    df_refs["text"] = df_refs["title_norm"].fillna("") + " " + df_refs["abstract"].fillna("")
    
    # 2. Tokenize Corpus
    log.info("Tokenizing corpus...")
    corpus_tokens = [simple_tokenize(text) for text in tqdm(df_refs["text"], desc="Tokenizing Docs")]
    
    # Map internal index back to ref_id
    corpus_ids = df_refs["ref_id"].values
    
    # 3. Build Index
    log.info("Building BM25 Index...")
    bm25 = BM25Okapi(corpus_tokens, k1=K1, b=B)
    log.info("Index ready.")

    # 4. Load Topics
    with open(JSON_IN, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    truth_map = {str(e.get("id")): [str(x).strip() for x in (e.get("ref_id") or [])] for e in entries}
    
    rows_for_df = []
    experiments = ["title", "objective", "combined"]

    # 5. Run Queries
    for entry in tqdm(entries, desc="Evaluating Queries"):
        sr_id = str(entry.get("id"))
        truth_ids = truth_map.get(sr_id, [])
        
        q_texts = {
            "title": (entry.get("sr_title") or "").strip(),
            "objective": (entry.get("objective") or "").strip(),
        }
        q_texts["combined"] = f"{q_texts['title']} {q_texts['objective']}"
        
        for q_type in experiments:
            raw_text = q_texts[q_type]
            tokenized_query = simple_tokenize(raw_text)
            
            if not tokenized_query:
                # Empty query
                rows_for_df.append({
                    "id": sr_id, "query_type": q_type,
                    "retrieved_count": 0, "truth_count": len(truth_ids),
                    "precision": 0.0, "recall": 0.0, "map": 0.0,
                    **{f"p@{k}": 0.0 for k in K_LEVELS},
                    **{f"r@{k}": 0.0 for k in K_LEVELS}
                })
                continue
            
            doc_scores = bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(doc_scores)[-RETRIEVAL_LIMIT:][::-1]
            retrieved_ids = corpus_ids[top_n_indices]
            
            metrics = evaluate(retrieved_ids, truth_ids)
            
            rows_for_df.append({
                "id": sr_id,
                "query_type": q_type,
                **metrics
            })

    # 6. Save
    scores_df = pd.DataFrame(rows_for_df)
    Path(OUTPUT_SCORES).parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(OUTPUT_SCORES, index=False)
    
    log.info(f"[OK] Saved Standard BM25 scores to {OUTPUT_SCORES}")
    
    # Summary
    log.info("\n=== Summary (Mean Scores) ===")
    summary = scores_df.groupby("query_type")[["precision", "recall", "map", "p@10", "r@100"]].mean()
    log.info(f"\n{summary}")