import pandas as pd
import igraph as ig
import leidenalg

PHASE2_NETWORK_FILE = "data/string_network_expanded_0.7.tsv"
OUTPUT_FILE = "results/lung_leiden_modules.csv"

# Load network
df = pd.read_csv(PHASE2_NETWORK_FILE, sep="\t")
df = df.dropna(subset=['combined_score'])
G = ig.Graph.TupleList(df[['protein1','protein2','combined_score']].itertuples(index=False), 
                       weights=True, directed=False)

# Leiden clustering
partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, weights='weight')
modules = {G.vs[i]['name']: part for i, part in enumerate(partition.membership)}

# Save
pd.DataFrame(list(modules.items()), columns=['gene','module']).to_csv(OUTPUT_FILE, index=False)
print("Leiden modules saved to", OUTPUT_FILE)