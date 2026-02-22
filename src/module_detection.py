import pandas as pd
import networkx as nx
import leidenalg
import igraph as ig
import community as community_louvain  # Louvain

# ----------------------------
# PARAMETERS
# ----------------------------
RWR_FILE = "results/rwr_top50_results.tsv"
NETWORK_FILE = "data/string_network_expanded_0.7.tsv"
RESULTS_LEIDEN_FILE = "results/leiden_modules.tsv"
RESULTS_LOUVAIN_FILE = "results/louvain_modules.tsv"

# ----------------------------
# LOAD NETWORK
# ----------------------------
df = pd.read_csv(NETWORK_FILE, sep="\t")
df = df[["protein1", "protein2", "combined_score"]]
df["combined_score"] = df["combined_score"] / 1000.0

G = nx.from_pandas_edgelist(df, source="protein1", target="protein2", edge_attr="combined_score")
print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ----------------------------
# LOAD TOP RWR PREDICTIONS
# ----------------------------
rwr_df = pd.read_csv(RWR_FILE, sep="\t")
top_genes = rwr_df["Gene"].tolist()

# Subgraph of top genes
G_sub = G.subgraph(top_genes).copy()
print(f"Subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

# ----------------------------
# LEIDEN COMMUNITY DETECTION
# ----------------------------
# Convert nx to igraph
mapping = {node: i for i, node in enumerate(G_sub.nodes())}
inv_mapping = {i: node for node, i in mapping.items()}
edges = [(mapping[u], mapping[v]) for u, v in G_sub.edges()]

g_igraph = ig.Graph(edges=edges, directed=False)
partition = leidenalg.find_partition(g_igraph, leidenalg.ModularityVertexPartition)

# Save modules
leiden_modules = []
for cid, nodes in enumerate(partition):
    for node in nodes:
        leiden_modules.append({"Gene": inv_mapping[node], "Module": cid})
leiden_df = pd.DataFrame(leiden_modules)
leiden_df.to_csv(RESULTS_LEIDEN_FILE, sep="\t", index=False)
print(f"Leiden modules saved to {RESULTS_LEIDEN_FILE}")

# ----------------------------
# LOUVAIN COMMUNITY DETECTION
# ----------------------------
louvain_partition = community_louvain.best_partition(G_sub, weight="combined_score")
louvain_df = pd.DataFrame(list(louvain_partition.items()), columns=["Gene", "Module"])
louvain_df.to_csv(RESULTS_LOUVAIN_FILE, sep="\t", index=False)
print(f"Louvain modules saved to {RESULTS_LOUVAIN_FILE}")