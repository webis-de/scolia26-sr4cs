"""
Evaluate LLM-generated SQLite queries against SQLite FTS5.
"""

import json
import re
import sqlite3
import traceback
from pathlib import Path
from typing import List, Tuple, Iterable, Set, Dict, Any

import pandas as pd
from tqdm import tqdm
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# =============== Config ===============
DB_PATH = "./data/refs.db"
DEBUG_SQL = False

JSON_WITH_REFINED_IN = "./data/sr4cs_llm_generated.json"
JSON_WITH_REFINED_OUT = "./data/sr4cs_llm_generated_with_sql.json"
OUTPUT_SCORES = "./data/results/llm_query_eval_scores.parquet"
LOG_FILE = "./logs/llm_query_eval.log"

K_LEVELS = [1, 5, 10, 20, 50, 100, 1000]
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
    order = f" ORDER BY bm25({table})" if use_bm25 else ""
    return f"SELECT {cols} FROM {table} WHERE {table} MATCH ?{order};"


def _fts_search_lit(raw_query: str, with_text: bool) -> List[Tuple]:
    con = _connect_ro(DB_PATH, debug=DEBUG_SQL)
    cur = con.cursor()

    # Attempt 1: Try with BM25 sorting
    sql = _select_sql("refs_fts_lit", with_text, use_bm25=True)
    try:
        rows = cur.execute(sql, (raw_query,)).fetchall()
        con.close()
        return rows
    except sqlite3.OperationalError as e:
        if DEBUG_SQL:
            print(f"[INFO ] bm25 ORDER BY not available: {e}")

    # Attempt 2: Fallback without sorting
    sql_fallback = _select_sql("refs_fts_lit", with_text, use_bm25=False)
    rows = cur.execute(sql_fallback, (raw_query,)).fetchall()
    con.close()
    return rows


def fts_search(raw_query: str, with_text: bool = True) -> Dict[str, Any]:
    rows_lit = _fts_search_lit(raw_query, with_text=with_text)
    return {
        "rows": rows_lit,
        "count_lit": len(rows_lit),
    }


# =============== Metrics ===============
def _to_str_set(ids: Iterable) -> Set[str]:
    return {str(x).strip() for x in ids if str(x).strip()}


def evaluate(retrieved_ids_ordered: List[str], truth_ids: Iterable) -> dict:
    R_set = _to_str_set(retrieved_ids_ordered)
    T_set = _to_str_set(truth_ids)
    
    tp_ids = R_set & T_set
    fp_ids = R_set - T_set
    fn_ids = T_set - R_set

    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    def f_beta(p: float, r: float, beta: float) -> float:
        if p == 0.0 and r == 0.0: return 0.0
        b2 = beta * beta
        denom = b2 * p + r
        return (1 + b2) * p * r / denom if denom else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f3": f_beta(precision, recall, 3.0),
        "tp": tp, "fp": fp, "fn": fn,
        "retrieved_count": len(R_set),
        "truth_count": len(T_set),
    }

    clean_retrieved = [str(x).strip() for x in retrieved_ids_ordered]
    hits_vector = []
    num_relevant = len(T_set)
    cumulative_hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(clean_retrieved):
        is_hit = 1 if doc_id in T_set else 0
        hits_vector.append(is_hit)
        if is_hit:
            cumulative_hits += 1
            sum_precisions += cumulative_hits / (i + 1)

    metrics["map"] = (sum_precisions / num_relevant) if num_relevant > 0 else 0.0

    for k in K_LEVELS:
        hits_at_k = hits_vector[:k]
        tp_at_k = sum(hits_at_k)
        metrics[f"p@{k}"] = tp_at_k / k
        metrics[f"r@{k}"] = (tp_at_k / num_relevant) if num_relevant > 0 else 0.0

    return metrics


# =============== Truth Helper ===============
def normalize_ref_ids(val) -> List[str]:
    if val is None: return []
    if isinstance(val, list): return [str(x).strip() for x in val]
    return [str(val).strip()]

def build_truth_map_from_entries(entries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    truth: Dict[str, List[str]] = {}
    for e in entries:
        sr_id = str(e.get("id", "")).strip()
        truth[sr_id] = normalize_ref_ids(e.get("ref_id"))
    return truth


# =============== Robust Sanitizer (FIXED) ===============
def _fix_unknown_columns(q: str) -> str:
    """Replaces hallucinations (analysis:, term:) with 'abstract:'."""
    valid_cols = {"title", "abstract"}
    def _repl(m):
        prefix = m.group(1)
        if prefix.lower() in valid_cols:
            return m.group(0)
        return "abstract:" 
    return re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*):', _repl, q)

def _fix_quoted_wildcards(q: str) -> str:
    """
    FIXED REGEX: Handles trailing wildcards correctly.
    Fixes LLM outputting "term*" (inside quotes) or "fault-toleran*".
    """
    def _rewrite(m):
        prefix = m.group(1) or ""
        content = m.group(2)
        
        # If it has a hyphen and a wildcard (the crasher)
        if "-" in content and "*" in content:
            # abstract:"fault-toleran*" -> abstract:(fault AND toleran*)
            parts = content.replace("-", " ").split()
            cleaned_parts = " AND ".join(parts)
            return f"{prefix}({cleaned_parts})"
            
        # If it has just a wildcard (e.g. "fail*") -> unquote it
        if "*" in content:
            return f"{prefix}{content}"
            
        # Otherwise keep quotes
        return m.group(0)

    # UPDATED REGEX: ([^\"]* instead of [^\"]+) allows wildcard at end of string
    return re.sub(r'([a-zA-Z0-9_]+:)?\"([^\"]*\*+[^\"]*)\"', _rewrite, q)

def _remove_near(s: str) -> str:
    s = re.sub(r"\bNEAR\s*/\s*\d+\b", "AND", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNEAR\b", "AND", s, flags=re.IGNORECASE)
    return s

def _drop_not_clauses(s: str) -> str:
    s = re.sub(r"\bAND\s+NOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNOT\s*\((?:[^()]*|\([^()]*\))*\)", "", s, flags=re.IGNORECASE)
    return s

def _cleanup_operators(q: str) -> str:
    q = " " + q + " "
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'\bOR\s+OR\b', 'OR', q, flags=re.IGNORECASE)
    q = re.sub(r'\bAND\s+AND\b', 'AND', q, flags=re.IGNORECASE)
    q = re.sub(r'\bAND\s+OR\b', 'OR', q, flags=re.IGNORECASE)
    q = re.sub(r'\bOR\s+AND\b', 'OR', q, flags=re.IGNORECASE)
    q = re.sub(r'\(\s*(?:OR|AND)\s+', '(', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+(?:OR|AND)\s*\)', ')', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*(?:OR|AND)\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\s+(?:OR|AND)\s*$', '', q, flags=re.IGNORECASE)
    return q.strip()

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

def normalize_query(q: str) -> str:
    s = q.strip()
    s = _fix_unknown_columns(s)
    s = _fix_quoted_wildcards(s)
    s = _remove_near(s)
    s = _drop_not_clauses(s)
    s = _balance_parens(s)
    s = _cleanup_operators(s)
    s = s.replace(" - ", " ") 
    return " ".join(s.split())

def quick_sanitize(q: str) -> str:
    return normalize_query(q)

def run_query_no_llm(raw_q: str) -> Dict[str, Any]:
    # 1) Try Raw
    try:
        return fts_search(raw_q, with_text=True)
    except sqlite3.OperationalError as e:
        log.error(f"RAW failed: {e}")

    # 2) Try Sanitized
    try:
        q1 = quick_sanitize(raw_q)
        # Log truncation to avoid huge log files
        log.info(f"Applying quick_sanitize → {q1[:100]}...")
        return fts_search(q1, with_text=True)
    except sqlite3.OperationalError as e:
        # FIXED: e is available here, but deleted after. 
        # We must Raise or Return here.
        log.error(f"SANITIZED failed: {e}")
        raise e  # Re-raise to be caught by the main loop


# =============== Main ===============
if __name__ == "__main__":
    log.info("=== Run started (LLM Query Eval) ===")
    
    with open(JSON_WITH_REFINED_IN, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)
    log.info(f"Loaded {len(entries)} SR entries.")

    truth_map = build_truth_map_from_entries(entries)
    
    Path(JSON_WITH_REFINED_OUT).parent.mkdir(parents=True, exist_ok=True)

    fixed_count = 0
    rows_for_df = []

    for i_entry, entry in enumerate(tqdm(entries, desc="Evaluating", unit="sr"), 1):
        sr_id = str(entry.get("id", "")).strip()
        
        # Handle String vs List
        raw_val = entry.get("llm_generated_query_sqlite")
        if isinstance(raw_val, str):
            refined_queries = [raw_val]
        elif isinstance(raw_val, list):
            refined_queries = raw_val
        else:
            refined_queries = []

        try:
            log.info(f"[SR {sr_id}] Start | n_queries={len(refined_queries)}")
            
            retrieved_ids_ordered: List[str] = []
            seen_ids: Set[str] = set()

            for qi, q in enumerate(refined_queries):
                q = (q or "").strip()
                if not q: continue
                
                try:
                    res = fts_search(q, with_text=True)
                except sqlite3.OperationalError:
                    # Attempt Repair
                    try:
                        res = run_query_no_llm(q)
                        fixed_count += 1
                        # Persist fix
                        sanitized_q = quick_sanitize(q)
                        if isinstance(entry.get("llm_generated_query_sqlite"), list):
                            entry["llm_generated_query_sqlite"][qi] = sanitized_q
                        elif isinstance(entry.get("llm_generated_query_sqlite"), str):
                            entry["llm_generated_query_sqlite"] = sanitized_q
                    except sqlite3.OperationalError:
                        log.error(f"Repair failed for SR {sr_id}. Skipping query.")
                        continue 

                rows = res["rows"]
                for ref_id, _t, _a in rows:
                    rid_str = str(ref_id).strip()
                    if rid_str not in seen_ids:
                        seen_ids.add(rid_str)
                        retrieved_ids_ordered.append(rid_str)

            # Evaluate
            truth_ids = truth_map.get(sr_id, [])
            metrics = evaluate(retrieved_ids_ordered, truth_ids)
            
            rows_for_df.append(
                {"id": sr_id, "n_queries": len(refined_queries), **metrics}
            )

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"[SR {sr_id}] CRITICAL ERROR: {e}\n{tb}")
            
            empty_metrics = {k: 0.0 for k in ["precision", "recall", "f1", "f3", "map", *[f"p@{x}" for x in K_LEVELS], *[f"r@{x}" for x in K_LEVELS]]}
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
    Path(OUTPUT_SCORES).parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(OUTPUT_SCORES, index=False)
    print(f"[OK] Wrote scores to {OUTPUT_SCORES} (deterministic_fixes={fixed_count})")