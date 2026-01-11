"""Evaluate refined SQLite queries against SQLite FTS5 and compute retrieval metrics (Set + Ranked)."""

import json
import re
import sqlite3
import traceback
from pathlib import Path
from typing import List, Tuple, Iterable, Set, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# =============== Config ===============
DB_PATH = "./data/refs.db"
DEBUG_SQL = False

JSON_WITH_REFINED_IN = "./data/sr4cs_with_sql.json"
JSON_WITH_REFINED_OUT = "./data/sr4cs_with_sql.json"
OUTPUT_SCORES = "./data/results/sqlite_eval_scores.parquet"
LOG_FILE = "./logs/sqlite_eval.log"

# Metrics Configuration
# We calculate P@k and R@k for these cutoffs
K_LEVELS = [1, 5, 10, 20, 50, 100, 1000]

# Deterministic repair only (no LLM)
CHECKPOINT_EVERY_SR = 50
CHECKPOINT_ON_FIXES = 10

# =============== Logging ===============
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
log = logging.getLogger("sqlite_eval")


# =============== SQLite ===============
def _connect_ro(db_path: str, debug: bool = DEBUG_SQL) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = 1;")
    except sqlite3.OperationalError:
        pass
    if debug:
        con.set_trace_callback(lambda s: print(f"[TRACE] {s}"))
    return con


def _select_sql(table: str, with_text: bool, use_bm25: bool) -> str:
    cols = "ref_id, title, abstract" if with_text else "ref_id"
    # IMPORTANT: This ORDER BY is critical for @k metrics.
    order = f" ORDER BY bm25({table})" if use_bm25 else ""
    return f"SELECT {cols} FROM {table} WHERE {table} MATCH ?{order};"


def _fts_search_lit(raw_query: str, with_text: bool) -> List[Tuple]:
    con = _connect_ro(DB_PATH, debug=DEBUG_SQL)
    cur = con.cursor()

    # Attempt 1: Try with BM25 sorting (Ranked)
    sql = _select_sql("refs_fts_lit", with_text, use_bm25=True)
    if DEBUG_SQL:
        print(f"\n[QUERY] table=refs_fts_lit (bm25 attempt)")
        print(f"[MATCH] {raw_query}")
        print(f"[SQL  ] {sql.strip()}")
    
    try:
        rows = cur.execute(sql, (raw_query,)).fetchall()
        con.close()
        return rows
    except sqlite3.OperationalError as e:
        if DEBUG_SQL:
            print(f"[INFO ] bm25 ORDER BY not available: {e}")
            print("[INFO ] Falling back to no ORDER BY")

    # Attempt 2: Fallback without sorting (Unranked/Arbitrary)
    # Note: Metrics @k will be unstable/random in this case.
    sql_fallback = _select_sql("refs_fts_lit", with_text, use_bm25=False)
    rows = cur.execute(sql_fallback, (raw_query,)).fetchall()
    con.close()
    return rows


def fts_search(raw_query: str, with_text: bool = True) -> Dict[str, Any]:
    """Search only the literal table (ES-like). Returns rows in rank order if possible."""
    rows_lit = _fts_search_lit(raw_query, with_text=with_text)
    return {
        "rows": rows_lit,
        "count_lit": len(rows_lit),
    }


# =============== Metrics (Set + Ranked) ===============
def _to_str_set(ids: Iterable) -> Set[str]:
    return {str(x).strip() for x in ids if str(x).strip()}


def evaluate(retrieved_ids_ordered: List[str], truth_ids: Iterable) -> dict:
    """
    Computes both Set metrics (Precision, Recall, F1) and Ranked metrics (P@k, R@k, MAP).
    Expects retrieved_ids_ordered to be a list preserving the ranking.
    """
    # 1. Prepare Sets for overlap checks
    R_set = _to_str_set(retrieved_ids_ordered)
    T_set = _to_str_set(truth_ids)
    
    # 2. Standard Set Metrics (Table 1)
    tp_ids = R_set & T_set
    fp_ids = R_set - T_set
    fn_ids = T_set - R_set

    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(p: float, r: float, beta: float) -> float:
        if p == 0.0 and r == 0.0:
            return 0.0
        b2 = beta * beta
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f3": f_beta(precision, recall, 3.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "retrieved_count": len(R_set),
        "truth_count": len(T_set),
    }

    # 3. Ranked Metrics (Table 2)
    # We iterate through the ordered list to compute @k
    # Clean inputs to ensure string matching works
    clean_retrieved = [str(x).strip() for x in retrieved_ids_ordered]
    
    hits_vector = [] # 1 if relevant, 0 if not
    num_relevant = len(T_set)
    cumulative_hits = 0
    sum_precisions = 0.0 # for MAP

    for i, doc_id in enumerate(clean_retrieved):
        is_hit = 1 if doc_id in T_set else 0
        hits_vector.append(is_hit)
        
        if is_hit:
            cumulative_hits += 1
            # Precision at this rank position (i+1)
            sum_precisions += cumulative_hits / (i + 1)

    # MAP (Average Precision for this query)
    # If no relevant docs exist, MAP is 0. If relevant docs exist but we found none, AP is 0.
    if num_relevant > 0:
        metrics["map"] = sum_precisions / num_relevant
    else:
        metrics["map"] = 0.0

    # Calculate @k metrics
    # Note: If k > len(retrieved), we treat missing ranks as non-relevant (standard IR practice)
    for k in K_LEVELS:
        # Slice the hits vector up to k
        hits_at_k = hits_vector[:k]
        tp_at_k = sum(hits_at_k)
        
        # P@k = relevant_in_top_k / k
        metrics[f"p@{k}"] = tp_at_k / k
        
        # R@k = relevant_in_top_k / total_relevant
        if num_relevant > 0:
            metrics[f"r@{k}"] = tp_at_k / num_relevant
        else:
            metrics[f"r@{k}"] = 0.0

    return metrics


# =============== Ground truth from JSON ===============
def normalize_ref_ids(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    return [str(val).strip()]


def build_truth_map_from_entries(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    truth: Dict[str, List[str]] = {}
    for e in entries:
        sr_id = str(e.get("id", "")).strip()
        truth[sr_id] = normalize_ref_ids(e.get("ref_id"))
    return truth


# =============== Minimal sanitizer ===============
def _needs_quotes(token: str) -> bool:
    return bool(re.search(r'[^a-z0-9*"]', token))


def _quote_fielded_terms(s: str) -> str:
    def _q(m: re.Match) -> str:
        fld, term = m.group(1), m.group(2)
        if term.startswith('"') and term.endswith('"') and len(term) >= 2:
            return f"{fld}:{term}"
        if _needs_quotes(term):
            return f'{fld}:"{term}"'
        return f"{fld}:{term}"

    return re.sub(rf"\b(title|abstract):([^\s()]+)", _q, s, flags=re.IGNORECASE)


def _remove_near(s: str) -> str:
    s = re.sub(r"\bNEAR\s*/\s*\d+\b", "AND", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNEAR\b", "AND", s, flags=re.IGNORECASE)
    return s


def _drop_not_clauses(s: str) -> str:
    s = re.sub(r"\bAND\s+NOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r'\bAND\s+NOT\s+(?:"[^"]+"|\([^)]+\)|\S+)', "", s, flags=re.IGNORECASE)
    s = re.sub(r'\bNOT\s+(?:"[^"]+"|\([^)]+\)|\S+)', "", s, flags=re.IGNORECASE)
    return s


def _balance_quotes(s: str) -> str:
    if s.count('"') % 2 == 1:
        s = s.rstrip()
        s = s[:-1] if s.endswith('"') else s + '"'
    return s


def _balance_parens(s: str) -> str:
    opens = s.count("(")
    closes = s.count(")")
    if opens > closes:
        s = s + (")" * (opens - closes))
    elif closes > opens:
        surplus = closes - opens
        for _ in range(surplus):
            pos = s.rfind(")")
            if pos >= 0:
                s = s[:pos] + s[pos + 1 :]
    return s


def _strip_trailing_boolean(s: str) -> str:
    return re.sub(r"\s*(?:OR|AND)\s*\)*\s*$", "", s, flags=re.IGNORECASE)


def normalize_query(q: str) -> str:
    s = q.strip()
    s = _remove_near(s)
    s = _drop_not_clauses(s)
    s = _quote_fielded_terms(s)
    s = _balance_quotes(s)
    s = _balance_parens(s)
    s = _strip_trailing_boolean(s)
    s = " ".join(s.split())
    return s


def quick_sanitize(q: str) -> str:
    return normalize_query(q)


# =============== Deterministic recovery ===============
def run_query_no_llm(raw_q: str) -> Dict[str, Any]:
    # 1) raw
    try:
        return fts_search(raw_q, with_text=True)
    except sqlite3.OperationalError as e:
        log.error(f"RAW failed: {e}")

    # 2) deterministic sanitize
    q1 = quick_sanitize(raw_q)
    if q1 != raw_q:
        log.info(f"Applying quick_sanitize → {q1[:240]}{' …' if len(q1)>240 else ''}")
        try:
            return fts_search(q1, with_text=True)
        except sqlite3.OperationalError as e:
            log.error(f"SANITIZED failed: {e}")

    # 3) last-resort normalize
    q_last = normalize_query(raw_q)
    log.info(f"Last-resort normalize → {q_last[:240]}{' …' if len(q_last)>240 else ''}")
    res = fts_search(q_last, with_text=True)
    return res


# =============== Main ===============
if __name__ == "__main__":
    log.info("=== Run started (Set + Ranked Metrics) ===")
    log.info(f"DB_PATH={DB_PATH}")
    log.info(f"OUTPUT_SCORES={OUTPUT_SCORES}")

    with open(JSON_WITH_REFINED_IN, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)
    log.info(f"Loaded {len(entries)} SR entries.")

    truth_map = build_truth_map_from_entries(entries)
    
    Path(JSON_WITH_REFINED_OUT).parent.mkdir(parents=True, exist_ok=True)

    fixed_count = 0
    rows_for_df = []

    for i_entry, entry in enumerate(tqdm(entries, desc="Evaluating", unit="sr"), 1):
        sr_id = str(entry.get("id", "")).strip()
        refined_queries: List[str] = entry.get("sqlite_refined_queries") or []

        try:
            log.info(f"[SR {sr_id}] Start | n_queries={len(refined_queries)}")
            
            # CRITICAL CHANGE: Use list to preserve order, seen_ids to dedup
            retrieved_ids_ordered: List[str] = []
            seen_ids: Set[str] = set()

            for qi, q in enumerate(refined_queries):
                q = (q or "").strip()
                if not q: continue

                try:
                    res = fts_search(q, with_text=True)
                except sqlite3.OperationalError:
                    res = run_query_no_llm(q)
                    fixed_count += 1
                    # Best-effort persist fix
                    try:
                        entry["sqlite_refined_queries"][qi] = quick_sanitize(q)
                    except: pass

                # Append results while maintaining order and uniqueness
                # (Standard approach: First query's results are top ranked)
                rows = res["rows"]
                for ref_id, _t, _a in rows:
                    rid_str = str(ref_id).strip()
                    if rid_str not in seen_ids:
                        seen_ids.add(rid_str)
                        retrieved_ids_ordered.append(rid_str)

            # Evaluate (Pass ordered list)
            truth_ids = truth_map.get(sr_id, [])
            metrics = evaluate(retrieved_ids_ordered, truth_ids)
            
            log.info(f"[SR {sr_id}] retrieved={metrics['retrieved_count']} P={metrics['precision']:.4f} R={metrics['recall']:.4f} MAP={metrics['map']:.4f}")
            
            rows_for_df.append(
                {"id": sr_id, "n_queries": len(refined_queries), **metrics}
            )

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"[SR {sr_id}] ERROR: {e}\n{tb}")
            # Empty metrics row
            empty_metrics = {k: 0.0 for k in ["precision", "recall", "f1", "f3", "map"]}
            # Add zeroed @k keys
            for k in K_LEVELS:
                empty_metrics[f"p@{k}"] = 0.0
                empty_metrics[f"r@{k}"] = 0.0
            
            rows_for_df.append({
                "id": sr_id, 
                "n_queries": len(refined_queries), 
                "retrieved_count": 0,
                "truth_count": len(truth_map.get(sr_id, [])),
                **empty_metrics
            })

        # Checkpoint
        if (i_entry % CHECKPOINT_EVERY_SR == 0) or (fixed_count and fixed_count % CHECKPOINT_ON_FIXES == 0):
            with open(JSON_WITH_REFINED_OUT, "w", encoding="utf-8") as wf:
                json.dump(entries, wf, ensure_ascii=False, indent=2)

    # Final save
    with open(JSON_WITH_REFINED_OUT, "w", encoding="utf-8") as wf:
        json.dump(entries, wf, ensure_ascii=False, indent=2)

    scores_df = pd.DataFrame(rows_for_df)
    scores_df.to_parquet(OUTPUT_SCORES, index=False)
    print(f"[OK] Wrote scores to {OUTPUT_SCORES} (deterministic_fixes={fixed_count})")