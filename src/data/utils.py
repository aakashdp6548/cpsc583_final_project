import torch

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