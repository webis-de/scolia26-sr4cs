import pandas as pd

# 1. Load Data
df = pd.read_parquet("./data/results/dense_baseline_scores.parquet")

# 2. Define Metrics
metrics = [
    'precision', 'recall', 'f1', 'f3', 'map', 
    'p@1', 'r@1', 'p@5', 'r@5', 'p@10', 'r@10', 
    'p@20', 'r@20', 'p@50', 'r@50', 'p@100', 'r@100', 'p@1000', 'r@1000'
]

# 3. Group by Query Type and Calculate Mean
grouped_means = df.groupby("query_type")[metrics].mean()

# 4. Print for Inspection
print(grouped_means.round(3))

# 5. Save to CSV
# This will create a CSV with 3 rows (combined, objective, title) and all metric columns
grouped_means.to_csv("./data/results/dense_macro_avg_by_type.csv")