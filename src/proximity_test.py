import networkx as nx
import pandas as pd
import numpy as np
import random
from collections import defaultdict

# ----------------------------
# LOAD DATA
# ----------------------------

# Load STRING network (edge list: geneA geneB weight)
df = pd.read_csv("data/string_network_expanded_0.7.tsv", sep="\t")

# Ensure correct column names
df = df[["protein1", "protein2", "combined_score"]]

# Convert STRING score (0–1000) to 0–1 scale
df["combined_score"] = df["combined_score"] / 1000.0

G = nx.from_pandas_edgelist(
    df,
    source="protein1",
    target="protein2",
    edge_attr="combined_score"
)

# Load seed genes
seed_df = pd.read_csv("data/seed_genes_lung.tsv")
seeds = set(seed_df["gene_symbol"])

# Load RWR results
rwr_df = pd.read_csv("results/rwr_results.tsv", sep="\t")

# Select TOP 100 novel genes (exclude seeds)
top_n = 50
predicted = [
    g for g in rwr_df.sort_values("Score", ascending=False)["Gene"]
    if g not in seeds and g in G.nodes()
][:top_n]

predicted = set(predicted)

# --------------------------------
# FILTER TO NETWORK NODES
# --------------------------------

seeds = seeds & set(G.nodes())
predicted = predicted & set(G.nodes())

print("Seeds in network:", len(seeds))
print("Predicted in network:", len(predicted))

# ----------------------------
# OBSERVED PROXIMITY
# ----------------------------

def avg_shortest_distance(setA, setB, graph):
    # Convert STRING confidence to distance
    G_weighted = graph.copy()
    for u, v, d in G_weighted.edges(data=True):
        weight = d.get("combined_score", 1.0)
        d["weight"] = 1.0 / weight

    lengths = nx.multi_source_dijkstra_path_length(
        G_weighted,
        setB,
        weight="weight"
    )

    distances = []
    for node in setA:
        if node in lengths:
            distances.append(lengths[node])

    return np.mean(distances)

D_obs = avg_shortest_distance(predicted, seeds, G)

# ----------------------------
# DEGREE MATCHING
# ----------------------------

degree_dict = dict(G.degree())

# Bin nodes by degree
degree_bins = defaultdict(list)
for node, deg in degree_dict.items():
    degree_bins[deg].append(node)

def sample_degree_matched(original_set):
    sampled = set()
    for gene in original_set:
        deg = degree_dict[gene]
        candidates = degree_bins[deg]
        sampled_gene = random.choice(candidates)
        sampled.add(sampled_gene)
    return sampled

# ----------------------------
# PERMUTATION TEST
# ----------------------------

n_perm = 1000
rand_distances = []

for i in range(n_perm):
    rand_set = sample_degree_matched(predicted)
    D_rand = avg_shortest_distance(rand_set, seeds, G)
    rand_distances.append(D_rand)

rand_distances = np.array(rand_distances)

# Empirical p-value
p_value = (np.sum(rand_distances <= D_obs) + 1) / (n_perm + 1)

# Z-score
z_score = (D_obs - np.mean(rand_distances)) / np.std(rand_distances)

# ----------------------------
# OUTPUT
# ----------------------------

print("Observed proximity:", D_obs)
print("Random mean:", np.mean(rand_distances))
print("Random std:", np.std(rand_distances))
print("Empirical p-value:", p_value)
print("Z-score:", z_score)