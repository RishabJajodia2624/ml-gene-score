import pandas as pd
import networkx as nx
import numpy as np

# ----------------------------
# PARAMETERS
# ----------------------------
SEED_FILE = "data/seed_genes_lung.tsv"  # change to CD/AD if needed
NETWORK_FILE = "data/string_network_expanded_0.7.tsv"
TOP_N_SEEDS = 50
TOP_N_PREDICTIONS = 50
RESULTS_RWR_FILE = "results/rwr_top50_results.tsv"
NUM_RANDOM = 1000
RESTART_PROB = 0.7
MAX_ITER = 100
TOL = 1e-6

# ----------------------------
# HELPER FUNCTION: Weighted Shortest Distance
# ----------------------------
def weighted_avg_shortest_distance(setA, setB, graph):
    """
    Average shortest distance from nodes in setA to setB using 1/edge_weight as distance.
    """
    G_weighted = graph.copy()
    for u, v, d in G_weighted.edges(data=True):
        weight = d.get("combined_score", 1.0)
        d["weight"] = 1.0 / weight

    lengths = nx.multi_source_dijkstra_path_length(G_weighted, setB, weight="weight")
    distances = [lengths.get(node, np.nan) for node in setA]
    return np.nanmean(distances)

# ----------------------------
# LOAD NETWORK
# ----------------------------
df = pd.read_csv(NETWORK_FILE, sep="\t")
df = df[["protein1", "protein2", "combined_score"]]
df["combined_score"] = df["combined_score"] / 1000.0  # scale 0-1

G = nx.from_pandas_edgelist(df, source="protein1", target="protein2", edge_attr="combined_score")
print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ----------------------------
# LOAD TOP SEEDS
# ----------------------------
seeds_df = pd.read_csv(SEED_FILE, sep="\t")
all_seeds = seeds_df["gene_symbol"].head(TOP_N_SEEDS).tolist()

# Filter seeds present in network
top_seeds = [s for s in all_seeds if s in G.nodes()]
missing = set(all_seeds) - set(top_seeds)
print(f"Top {len(top_seeds)} seeds selected (filtered for network)")
if missing:
    print(f"Warning: {len(missing)} seeds not in network and skipped: {missing}")

# ----------------------------
# RANDOM WALK WITH RESTART (RWR)
# ----------------------------
def run_rwr(graph, seeds, restart_prob=RESTART_PROB, max_iter=MAX_ITER, tol=TOL):
    nodes = list(graph.nodes())
    idx_map = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)

    # adjacency matrix with normalized weights
    A = nx.to_numpy_array(graph, nodelist=nodes, weight="combined_score")
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1
    W = A / col_sums  # column stochastic

    # initial probability vector
    p0 = np.zeros(N)
    for s in seeds:
        if s in idx_map:
            p0[idx_map[s]] = 1.0 / len(seeds)
    p = p0.copy()

    # iteration
    for _ in range(max_iter):
        p_new = (1 - restart_prob) * np.dot(W, p) + restart_prob * p0
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new

    return dict(zip(nodes, p))

# ----------------------------
# RUN RWR
# ----------------------------
rwr_scores = run_rwr(G, top_seeds)
rwr_sorted = sorted(rwr_scores.items(), key=lambda x: x[1], reverse=True)
top_predictions = rwr_sorted[:TOP_N_PREDICTIONS]

# save top predictions
top_pred_df = pd.DataFrame(top_predictions, columns=["Gene", "Score"])
top_pred_df.to_csv(RESULTS_RWR_FILE, sep="\t", index=False)
print(f"Top {TOP_N_PREDICTIONS} RWR predictions saved to {RESULTS_RWR_FILE}")

# ----------------------------
# WEIGHTED PROXIMITY TEST
# ----------------------------
pred_genes = list(top_pred_df["Gene"])
observed = weighted_avg_shortest_distance(pred_genes, top_seeds, G)

degrees = dict(G.degree())
all_nodes = set(G.nodes()) - set(top_seeds)
rand_means = []

for _ in range(NUM_RANDOM):
    rand_nodes = np.random.choice(list(all_nodes), len(pred_genes), replace=False)
    rand_means.append(weighted_avg_shortest_distance(rand_nodes, top_seeds, G))

rand_mean = np.mean(rand_means)
rand_std = np.std(rand_means)
z_score = (observed - rand_mean) / rand_std
emp_p = (np.sum(np.array(rand_means) >= observed) + 1) / (NUM_RANDOM + 1)

print("\nWeighted Proximity Results:")
print(f"Observed proximity: {observed}")
print(f"Random mean: {rand_mean}")
print(f"Random std: {rand_std}")
print(f"Empirical p-value: {emp_p}")
print(f"Z-score: {z_score}")