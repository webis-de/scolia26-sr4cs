"""
Generate 'Zero-Shot' Boolean queries using Azure OpenAI.
Simulates a researcher asking an LLM to write a query based on Title + Objective.
Directly outputs SQLite FTS5 compatible syntax.
"""

import os
import json
import time
import logging
from typing import List, Dict
from pathlib import Path

from tqdm import tqdm
from openai import AzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# ========================
# Config
# ========================
# Reuse your existing credentials
AZURE_ENDPOINT = "https://ai-dsiplayground101757747291.cognitiveservices.azure.com/"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = "2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-mini"

# Input/Output
JSON_IN = Path("./data/sr4cs.json")
# We will save to a new file to avoid overwriting your manual translation work
OUTPUT_JSON = Path("./data/sr4cs_llm_generated.json")
LOG_FILE = Path("./logs/llm_gen.log")

CHECKPOINT_EVERY = 20
MAX_COMPLETION_TOKENS = 4000
TEMPERATURE = 0.0 # Deterministic generation
RETRIES = 3
BACKOFF_BASE = 2.0
SKIP_DONE = True

# ========================
# Logging
# ========================
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
logger = logging.getLogger("llm_gen")

# ========================
# Azure client
# ========================
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

# ========================
# Prompts
# ========================

SYSTEM_PROMPT = """You are an expert Information Retrieval Specialist for Computer Science Systematic Reviews. 
Your task is to write a high-recall, high-precision Boolean query based on a review's Title and Objective.
CRITICAL: You must output the query in strict SQLite FTS5 format."""

# We combine the "Generation" instruction with your existing "Syntax" constraints
USER_PROMPT_TEMPLATE = r"""
    <task>
    Generate a single Boolean search query for a Systematic Review based on the following topic.
    
    <topic_info>
    Title: {title}
    Objective: {objective}
    </topic_info>

    <constraints>
    1. WRITE A QUERY that captures the core concepts for this Review. Use appropriate synonyms and variations.
    2. FORMAT TARGET: SQLite FTS5 "MATCH" syntax.
    3. SYNTAX RULES:
       - Use ONLY fields title: and abstract:.
       - REPEAT the prefix for every term. Example: (title:AI OR abstract:AI).
       - AVOID using `title:(A OR B)`. ALWAYS use `(title:A OR title:B)`.
       - Use standard uppercase AND / OR / NOT.
       - Use `*` for truncation (e.g., `title:optimiz*`).
       - NO "NEAR" operators.
       - NO unquoted hyphens. Use quotes for phrases: `title:"machine learning"`.
    </constraints>

    <example>
    Input Title: Deep Learning Approaches for Energy-Efficient Computing
    Input Objective: To identify and evaluate deep learning techniques that enhance energy efficiency in computing systems.
    Output:
    (title:"deep learning" OR abstract:"deep learning" OR title:DL OR abstract:DL) AND 
    (title:"energy efficient" OR abstract:"energy efficient" OR title:"energy-efficient" OR abstract:"energy-efficient" OR title:energ* OR abstract:energ*) AND
    (title:comput* OR abstract:comput* OR title:system* OR abstract:system*)
    </example>

    <output_instruction>
    Return ONLY the raw query string. No markdown, no explanations.
    </output_instruction>
    </task>
""".strip()


def run_chat(system_prompt: str, user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        out = (resp.choices[0].message.content or "").strip()
        
        # Strip code fences if the LLM adds them
        if "```" in out:
            out = out.replace("```sql", "").replace("```", "").strip()
            
        return out
    except Exception as e:
        raise e


def generate_query(title: str, objective: str) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, objective=objective)
    
    for attempt in range(1, RETRIES + 1):
        try:
            return run_chat(SYSTEM_PROMPT, user_prompt)
        except (RateLimitError, APITimeoutError, APIError, Exception) as e:
            wait = BACKOFF_BASE ** (attempt - 1)
            logger.warning(f"Error (attempt {attempt}): {e}. Backing off {wait}s")
            time.sleep(wait)
            
    logger.error("Failed to generate query.")
    return ""


if __name__ == "__main__":
    # Load Data
    if OUTPUT_JSON.exists():
        logger.info(f"Resuming from {OUTPUT_JSON}")
        with OUTPUT_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.info(f"Starting fresh from {JSON_IN}")
        with JSON_IN.open("r", encoding="utf-8") as f:
            data = json.load(f)

    logger.info(f"Loaded {len(data)} entries.")
    
    processed_count = 0

    with tqdm(total=len(data), desc="Generating Queries") as pbar:
        for idx, entry in enumerate(data):
            entry_id = entry.get("id")
            
            # Skip if already done
            if SKIP_DONE and entry.get("llm_generated_query_sqlite"):
                pbar.update(1)
                continue

            # Prepare Input
            title = entry.get("sr_title", "")
            objective = entry.get("objective", "")
            
            # Generate
            logger.info(f"Generating for ID={entry_id}...")
            generated_q = generate_query(title, objective)
            
            # Save strictly to a new field
            entry["llm_generated_query_sqlite"] = generated_q
            
            processed_count += 1
            pbar.update(1)

            # Checkpoint
            if processed_count % CHECKPOINT_EVERY == 0:
                with OUTPUT_JSON.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("Checkpoint saved.")

    # Final Save
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] LLM Generation Complete. Saved to {OUTPUT_JSON}")