import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import torch.nn.functional as F

def plot_embeddings(embeddings, cell_types, save_path=None, epoch=None):
    """Plot TSNE visualization of embeddings colored by cell type.
    
    Args:
        embeddings (torch.Tensor): [N x D] embedding matrix
        cell_types (torch.Tensor): [N,] cell type indices
        save_path (str, optional): Path to save the plot
        epoch (int, optional): Epoch number
    """
    # Convert embeddings to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Compute TSNE
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings_np)

    cell_type_indices = torch.argmax(cell_types, dim=1).cpu().numpy()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    unique_cell_types = np.unique(cell_type_indices)
    for cell_type_idx in unique_cell_types:
        mask = (cell_type_indices == cell_type_idx)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   alpha=0.6, label=f"Cell Type {cell_type_idx}")
    plt.legend(title="Cell Type", loc="center left", bbox_to_anchor=(1, 0.5))
    if epoch is not None:
        plt.title(f"Cell Embeddings - Epoch {epoch}")
    else:
        plt.title("Cell Embeddings")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_loss_curves(losses, save_path=None):
    """Plot training loss curves.
    
    Args:
        losses (dict): Dictionary containing lists of losses
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    for name, values in losses.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_expression_matrix(expression_matrix, save_path=None):
    """Plot expression matrix as a heatmap.
    
    Args:
        expression_matrix (torch.Tensor): [N x G] expression matrix
        cell_type_labels (torch.Tensor): [N,] cell type indices
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(expression_matrix.cpu().numpy(), cmap="viridis")
    plt.title("Expression Matrix Heatmap")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_spatial_coords(spatial_coords, cell_types, save_path=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=cell_types.argmax(dim=1).cpu().numpy())
    # Add a legend for the cell type
    plt.legend(*scatter.legend_elements(), title="Cell Types")
    plt.title("Spatial Coordinates")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()