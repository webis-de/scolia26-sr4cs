# SR4CS — Systematic Review Test Collection for Computer Science

### Overview

SR4CS is a benchmark dataset of systematic reviews in Computer Science. This repository provides the full codebase for dataset construction and baseline experiments.
- The dataset itself (JSON + reference pool in multiple formats) is hosted on Zenodo: [LINK](https://zenodo.org/records/17163932)

This repository contains the pipelines for:
- Retrieving and filtering candidate SRs from DBLP.
- Parsing PDFs into structured text.
- Extracting search methodology fields with LLMs.
- Extracting and enriching references (metadata + abstracts).
- Building final JSON/Parquet/SQLite/Elasticsearch datasets.
- Translating Boolean Queries to SQL Match Syntax.
- Running baseline retrieval experiments (SQLite FTS5, BM25, Dense).

---

### Repository Layout
- src/retrieval/ — Fetch, filter, and prepare SR candidates + PDFs.
- src/extraction/ — OCR/LLM field extraction and reference parsing + enrichment.
- src/utils/ — Assembly and dataset hygiene (ID updates, metadata integration).
- src/experiments/ — Retrieval baselines (SQLite, BM25, Dense).

---

### Installation
- Python 3.10+
- Install deps:
    `pip install -r requirements.txt`

- External tools/services:
  - Grobid (reference extraction, requires local service).
  - AnyStyle (citation parsing).
  - Azure OpenAI (for LLM field/query extraction).
  - OCR model (Nanonets-OCR-s via VLLM).

---

### Main Steps 

1) Download SR candidates from DBLP (year‑sliced)

```
python src/retrieval/fetch_dblp_query_data.py
```

2) Filter to likely SRs (peer‑reviewed, OA, has DOI)

```
python src/retrieval/filter_retrieved_data.py
```

3) Resolve PDF links and download PDFs

```
python src/retrieval/get_pdf_link.py
python src/retrieval/download_pdfs.py
```

4) Parse SR full texts to Markdown

```
python src/extraction/nanonets_ocr.py
```

5) Extract SR search fields with LLM

```
python src/extraction/llm_extract.py
```

6) Extract references and build SR→ref mapping

```
python src/extraction/refs/ref_extract.py
```

7) Reference metadata enrichment

```
python src/extraction/refs/enrich_refs.py
# and/or: doi_based_fetch.py, europe_pmc_doi.py, title_based_fetch.py, arxiv_based_fetch.py
```

8) References Combination and mapping
```
python src/extraction/refs/parse_ref_to_csv.py
python src/extraction/refs/refs_mapping.py
python src/extraction/refs/final_filter_and_combine.py
```


9) Finalize SR JSON and clean ref lists

```
python src/utils/add_metadata.py
python src/utils/update_ref_ids.py
```

10) Translate queries, index references, and run experiments

```
# Query translation → adds sqlite_refined_queries
python src/experiments/transform_to_sqlite_query.py

# SQLite FTS5 index over refs and evaluation
python src/experiments/sqlite_build_fts5.py
python src/experiments/sqlite_query_all.py

# (Optional) Elasticsearch index and evaluation
python src/experiments/es_load_docs.py
python src/experiments/es_query_all.py
```
