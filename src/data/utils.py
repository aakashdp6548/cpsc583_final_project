import torch
import scanpy as sc
import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATASET_MAPPING = {
    "lymph_node": "V1_Human_Lymph_Node",
    "breast_cancer": "V1_Breast_Cancer_Block_A_Section_1",
}

def create_synthetic_data(num_cells=100, num_genes=1000, cell_type_idx=0, num_cell_types=3):
    """Create synthetic data for testing the model.
    
    Args:
        num_cells (int): Number of cells
        num_genes (int): Number of genes
        cell_type_idx (int): Index of the cell type for this dataset
        num_cell_types (int): Number of cell types
    """
    # Generate random gene expression matrix
    gene_expression = torch.randn(num_cells, num_genes)
    
    # Generate random spatial coordinates from a separate Gaussian for each cell type
    mean = torch.rand(2)
    cov = torch.eye(2) * 0.02
    spatial_coords = torch.distributions.MultivariateNormal(mean, cov).sample((num_cells,))
    spatial_coords = spatial_coords - spatial_coords.min()
    spatial_coords = spatial_coords / spatial_coords.max()  # Normalize to [0, 1]
    
    # Generate cell types
    cell_types = torch.zeros(num_cells, num_cell_types)
    cell_types[:, cell_type_idx] = 1
    
    return {
        "gene_expression": gene_expression,
        "spatial_coords": spatial_coords,
        "cell_types": cell_types,
    }


def create_combined_dataset(num_cells, num_genes, num_cell_types):
    """Create and combine synthetic datasets for multiple cell types.
    
    Args:
        num_cells (int): Number of cells per cell type
        num_genes (int): Number of genes
        num_cell_types (int): Number of cell types
    """
    total_cells = num_cells * num_cell_types
    gene_expression = []
    spatial_coords = []
    cell_types = []
    
    # Create one dataset for each tissue type
    for i in range(num_cell_types):
        data = create_synthetic_data(
            num_cells=num_cells,
            num_genes=num_genes,
            cell_type_idx=i,
            num_cell_types=num_cell_types
        )
        gene_expression.append(data["gene_expression"])
        spatial_coords.append(data["spatial_coords"])
        cell_types.append(data["cell_types"])
    
    # Randomly shuffle the cells to mix cell types
    indices = torch.randperm(total_cells)
    return {
        "gene_expression": torch.cat(gene_expression)[indices],
        "spatial_coords": torch.cat(spatial_coords)[indices],
        "cell_types": torch.cat(cell_types)[indices],
    } 


def construct_interaction_matrix(gene_ids, interactions):
    """Construct an interaction matrix from a list of gene ids and a list of interactions.
    
    Args:
        gene_ids (list): List of gene ids
        interactions (pd.DataFrame): DataFrame of interactions

    Returns:
        scipy.sparse.csr_matrix: Interaction matrix
    """
    interaction_matrix = torch.zeros((len(gene_ids), len(gene_ids)))
    for _, row in interactions.iterrows():
        source_gene = row["ensembl_a"]
        target_gene = row["ensembl_b"]
        if source_gene in gene_ids and target_gene in gene_ids:
            interaction_matrix[gene_ids.index(source_gene), gene_ids.index(target_gene)] = 1
    return interaction_matrix


def load_data(dataset_name, n_top_genes=2000):
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Dataset {dataset_name} not found")
    adata_path = f"data/{DATASET_MAPPING[dataset_name]}/{dataset_name}.h5ad"
    adata = anndata.read_h5ad(adata_path)

    # Subset to highly variable genes
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
    
    # Load gene interactions
    interactions = pd.read_csv(f"data/gene_interactions.csv")
    gene_ids = adata.var.gene_ids.tolist()  # Use ensembl ids
    interaction_matrix = construct_interaction_matrix(gene_ids, interactions)

    # Get cell type one-hot encoded
    num_cell_types = len(adata.obs["leiden"].cat.categories)
    cell_types = torch.zeros(len(adata.obs["leiden"]), num_cell_types)
    cell_types[torch.arange(len(adata.obs["leiden"])), adata.obs["leiden"].cat.codes.values] = 1
    
    return {
        "gene_expression": torch.tensor(adata.X.toarray()).float(),
        "spatial_coords": torch.tensor(adata.obsm["spatial"]).float(),
        "cell_types": cell_types.float(),
        "interaction_matrix": interaction_matrix.float(),
    }

if __name__ == "__main__":
    dataset = load_data("lymph_node")
    print(dataset["gene_expression"].shape)
    print(dataset["cell_types"].shape)
    print(dataset["interaction_matrix"].shape)