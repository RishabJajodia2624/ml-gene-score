import requests
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time

STRING_API_URL = "https://string-db.org/api/tsv/network"
SPECIES = 9606
SCORE_THRESHOLD = 700  # 0.7 confidence

print("Loading seed genes...")
seeds = pd.read_csv("../data/seed_genes_lung.tsv", sep="\t")
gene_list = seeds["gene_symbol"].dropna().unique().tolist()

print(f"Total seed genes: {len(gene_list)}")

edges = []
seen_edges = set()

print("Expanding network via STRING...")

for gene in tqdm(gene_list):
    params = {
        "identifiers": gene,
        "species": SPECIES,
        "required_score": SCORE_THRESHOLD
    }

    response = requests.post(STRING_API_URL, data=params)

    if response.status_code != 200:
        continue

    lines = response.text.strip().split("\n")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue

        p1 = parts[2]
        p2 = parts[3]
        score = float(parts[5])

        edge = tuple(sorted((p1, p2)))
        if edge not in seen_edges:
            seen_edges.add(edge)
            edges.append((p1, p2, score))

    time.sleep(0.2)  # prevent API throttling

print(f"\nTotal edges collected: {len(edges)}")

df = pd.DataFrame(edges, columns=["protein1", "protein2", "combined_score"])
df.to_csv("../data/string_network_expanded_0.7.tsv", sep="\t", index=False)

print("Saved expanded network.")

# Build graph
G = nx.Graph()
for p1, p2, score in edges:
    G.add_edge(p1, p2, weight=score)

print(f"Network nodes: {G.number_of_nodes()}")
print(f"Network edges: {G.number_of_edges()}")

seed_set = set(gene_list)
present_seeds = seed_set.intersection(set(G.nodes()))
print(f"Seed genes present in expanded network: {len(present_seeds)}")