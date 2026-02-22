import pandas as pd

results = pd.read_csv("rwr_results.tsv", sep="\t")
seeds = pd.read_csv("data/seed_genes_lung.tsv", sep="\t")

seed_set = set(seeds.iloc[:, 0])

# Remove seed genes
novel = results[~results['Gene'].isin(seed_set)]

top20_novel = novel.head(20)

print("Top 20 novel candidate genes:")
print(top20_novel)