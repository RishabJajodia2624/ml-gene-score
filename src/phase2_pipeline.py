# Import libraries
import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig
import leidenalg
import random
from tqdm import tqdm

# ---------- Step 1: Load data ----------
seed_genes = pd.read_csv("../data/seed_genes_lung.tsv", header=None)[0].tolist()
string_edges = pd.read_csv("../data/lung_STRING_network.csv")  # columns: geneA, geneB, combined_score
G = nx.from_pandas_edgelist(string_edges, 'geneA', 'geneB', edge_attr='combined_score')

# ---------- Step 2: Expand network ----------
present_seeds = [g for g in seed_genes if g in G.nodes]
neighbors = set()
for gene in present_seeds:
    neighbors.update(G.neighbors(gene))
expanded_nodes = set(present_seeds).union(neighbors)
G_expanded = G.subgraph(expanded_nodes).copy()

print(f"Expanded network: {G_expanded.number_of_nodes()} nodes, {G_expanded.number_of_edges()} edges")

# ---------- Step 3: RWR propagation ----------
def rwr(G, seeds, restart_prob=0.3, max_iter=100, tol=1e-6):
    nodes = list(G.nodes)
    idx_map = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1
    W = A / deg[:, None]
    p0 = np.zeros(N)
    for s in seeds:
        if s in idx_map:
            p0[idx_map[s]] = 1
    p0 /= p0.sum()
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1 - restart_prob) * W.T @ p + restart_prob * p0
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new
    return dict(zip(nodes, p))

rwr_scores = rwr(G_expanded, present_seeds)

# ---------- Step 4: Proximity significance ----------
actual_distances = {}
for n in G_expanded.nodes:
    dists = [nx.shortest_path_length(G_expanded, n, s) for s in present_seeds if nx.has_path(G_expanded, n, s)]
    actual_distances[n] = np.mean(dists) if dists else np.inf

def degree_preserving_null(G, seeds, n_iter=100):
    nodes = list(G.nodes)
    scores_null = {n: [] for n in nodes}
    for _ in tqdm(range(n_iter)):
        random_nodes = random.sample(nodes, len(seeds))
        for n in nodes:
            dists = [nx.shortest_path_length(G, n, r) for r in random_nodes if nx.has_path(G, n, r)]
            scores_null[n].append(np.mean(dists) if dists else np.inf)
    return scores_null

null_distances = degree_preserving_null(G_expanded, present_seeds, n_iter=100)
p_values = {n: sum(nd <= actual_distances[n] for nd in null_distances[n]) / len(null_distances[n]) for n in G_expanded.nodes}

# ---------- Step 5: Module detection ----------
ig_G = ig.Graph.TupleList(G_expanded.edges(), weights='combined_score', directed=False)
partition = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition)
modules = {}
for i, cluster in enumerate(partition):
    for node_idx in cluster:
        modules[ig_G.vs[node_idx]['name']] = i

# ---------- Step 6: Save results ----------
results = pd.DataFrame({
    'gene': list(G_expanded.nodes),
    'rwr_score': [rwr_scores[n] for n in G_expanded.nodes],
    'p_value': [p_values[n] for n in G_expanded.nodes],
    'module': [modules[n] for n in G_expanded.nodes]
})
results.to_csv("../results/lung_gene_modules_phase2.csv", index=False)
print("Phase 2 complete — results saved in results/lung_gene_modules_phase2.csv")

