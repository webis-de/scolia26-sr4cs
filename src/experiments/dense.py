"""
Benchmark Dense Retrieval using Sentence-Transformers (SBERT).
Baseline Model: all-MiniLM-L6-v2
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Set, Iterable
from tqdm import tqdm
import logging

# New Dependency
from sentence_transformers import SentenceTransformer, util

# =============== Config ===============
# Inputs
REFS_PARQUET = "./data/refs.parquet"
JSON_IN = "./data/sr4cs.json"

# Outputs
OUTPUT_SCORES = "./data/results/dense_baseline_scores.parquet"
LOG_FILE = "./logs/dense_eval.log"

# Model Config
# 'all-MiniLM-L6-v2' is the standard efficient baseline for semantic search
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 128  # Adjust based on GPU VRAM

# Retrieval Config
RETRIEVAL_LIMIT = 1000
K_LEVELS = [1, 5, 10, 20, 50, 100, 1000]

# =============== Logging ===============
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
log = logging.getLogger("dense_eval")


# =============== Metrics (Exact Copy to ensure consistency) ===============
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
    log.info(f"=== Dense Retrieval ({MODEL_NAME}) Run Started ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # 1. Load Model
    log.info(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 2. Load Corpus
    log.info(f"Loading corpus from {REFS_PARQUET}...")
    df_refs = pd.read_parquet(REFS_PARQUET, columns=["ref_id", "title_norm", "abstract"])
    
    # --- CRITICAL FIX START (ID Alignment) ---
    df_refs["ref_id"] = pd.to_numeric(df_refs["ref_id"], errors='coerce')
    df_refs = df_refs.dropna(subset=["ref_id"])
    df_refs["ref_id"] = df_refs["ref_id"].astype(int).astype(str)
    # --- CRITICAL FIX END ---
    
    # Prepare text for embedding
    df_refs["text"] = df_refs["title_norm"].fillna("") + " " + df_refs["abstract"].fillna("")
    
    # Map: Corpus Index -> Ref ID
    corpus_ids = df_refs["ref_id"].values
    corpus_sentences = df_refs["text"].tolist()

    # 3. Encode Corpus
    log.info(f"Encoding {len(corpus_sentences)} documents...")
    # This returns a Tensor on the computed device (usually GPU)
    # convert_to_tensor=True is crucial for fast semantic_search later
    corpus_embeddings = model.encode(
        corpus_sentences, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        normalize_embeddings=True # Normalizing allows DotProduct to act as Cosine Similarity
    )
    
    log.info(f"Corpus encoded. Shape: {corpus_embeddings.shape}")

    # 4. Load Topics
    with open(JSON_IN, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    truth_map = {str(e.get("id")): [str(x).strip() for x in (e.get("ref_id") or [])] for e in entries}
    
    rows_for_df = []
    experiments = ["title", "objective", "combined"]

    # 5. Run Queries
    log.info("Starting Query Evaluation Loop...")
    
    for entry in tqdm(entries, desc="Evaluating Queries"):
        sr_id = str(entry.get("id"))
        truth_ids = truth_map.get(sr_id, [])
        
        q_texts = {
            "title": (entry.get("sr_title") or "").strip(),
            "objective": (entry.get("objective") or "").strip(),
        }
        q_texts["combined"] = f"{q_texts['title']} {q_texts['objective']}"
        
        for q_type in experiments:
            query_text = q_texts[q_type]
            
            if not query_text:
                # Empty query -> 0 scores
                rows_for_df.append({
                    "id": sr_id, "query_type": q_type,
                    "retrieved_count": 0, "truth_count": len(truth_ids),
                    "precision": 0.0, "recall": 0.0, "map": 0.0,
                    **{f"p@{k}": 0.0 for k in K_LEVELS},
                    **{f"r@{k}": 0.0 for k in K_LEVELS}
                })
                continue
            
            # Encode Query (Single)
            query_embedding = model.encode(
                query_text, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            
            # Search
            # Returns: List[List[Dict{'corpus_id', 'score'}]]
            # We have 1 query, so we take results[0]
            hits = util.semantic_search(
                query_embedding, 
                corpus_embeddings, 
                top_k=RETRIEVAL_LIMIT,
                score_function=util.dot_score # Since normalized, this is Cosine
            )
            
            top_hits = hits[0]
            
            # Map Indices -> Ref IDs
            retrieved_ids = [corpus_ids[hit['corpus_id']] for hit in top_hits]
            
            # Evaluate
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
    
    log.info(f"[OK] Saved Dense Retrieval scores to {OUTPUT_SCORES}")
    
    # Summary
    log.info("\n=== Summary (Mean Scores) ===")
    summary = scores_df.groupby("query_type")[["precision", "recall", "map", "p@10", "r@100"]].mean()
    log.info(f"\n{summary}")