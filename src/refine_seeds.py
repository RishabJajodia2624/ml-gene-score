import pandas as pd

# Load original seeds (Phase 1B)
seeds_df = pd.read_csv("data/seed_genes_lung.tsv", sep="\t")

# Assuming your file has 'Gene' and a 'Score' or 'p-value' column. 
# Sort to get the most significant 50. 
# If sorting by p-value, use ascending=True. If by confidence, use ascending=False.
top_seeds = seeds_df.sort_values(by=seeds_df.columns[1], ascending=False).head(50)

# Save as new high-confidence seed set
top_seeds.to_csv("data/seed_genes_lung_top50.tsv", sep="\t", index=False)

print(f"Refinement Complete: Reduced seeds from {len(seeds_df)} to {len(top_seeds)}.")
