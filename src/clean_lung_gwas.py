import pandas as pd

P_THRESHOLD = 5e-8

print("Loading lung cancer GWAS file...")
df = pd.read_csv("/mnt/c/Users/risha/Downloads/lung_cancer_gwas.tsv",
                 sep="\t",
                 low_memory=False)

print("Filtering lung-related traits...")
lung_df = df[
    (df["DISEASE/TRAIT"].str.contains("lung", case=False, na=False)) |
    (df["MAPPED_TRAIT"].str.contains("lung", case=False, na=False))
]

print("Filtering genome-wide significant associations...")
lung_df = lung_df[lung_df["P-VALUE"] <= P_THRESHOLD]

print(f"Total significant lung associations: {len(lung_df)}")

def extract_genes(dataframe):
    genes = set()
    for col in ["REPORTED GENE(S)", "MAPPED_GENE"]:
        if col in dataframe.columns:
            for entry in dataframe[col].dropna():
                for g in str(entry).replace(" - ", ",").split(","):
                    g = g.strip()
                    if g and g != "NR":
                        genes.add(g)
    return sorted(list(genes))

genes = extract_genes(lung_df)

pd.DataFrame({"gene_symbol": genes}).to_csv(
    "../data/seed_genes_lung.tsv",
    sep="\t",
    index=False
)

print(f"Total unique lung cancer genes: {len(genes)}")
print("Saved as seed_genes_lung.tsv")