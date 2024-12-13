import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class BaseGNN(nn.Module):
    """Base class for GNN models."""
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Skip connection from gene expression to final embeddings
        self.gene_to_embedding = nn.Linear(hidden_dim, embedding_dim)
        
        # Graph attention layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, embedding_dim, heads=1)
        
        self.final_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Interaction predictor
        self.interaction_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = dropout
    
    def encode(self, x, edge_index):
        """Encode node features into embeddings."""
        # Encode gene expression
        h_genes = self.gene_encoder(x)
        gene_embeddings = self.gene_to_embedding(h_genes)
        
        # Message passing
        h = self.conv1(h_genes, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        graph_embeddings = self.conv2(h, edge_index)
        
        # Combine gene expression and graph embeddings
        combined = torch.cat([gene_embeddings, graph_embeddings], dim=1)
        final_embeddings = self.final_projection(combined)
        
        return final_embeddings
    
    def predict_interactions(self, embeddings):
        """Predict interaction scores from embeddings."""
        N = embeddings.size(0)
        # Create all pairs of embeddings
        emb_i = embeddings.unsqueeze(1).repeat(1, N, 1)  # [N x N x D]
        emb_j = embeddings.unsqueeze(0).repeat(N, 1, 1)  # [N x N x D]
        # Concatenate pairs
        pair_embeddings = torch.cat([emb_i, emb_j], dim=-1)  # [N x N x 2D]
        # Predict interaction scores
        return self.interaction_predictor(pair_embeddings).squeeze(-1)  # [N x N]


class InteractionGNN(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_cell_types,
        gene_interactions,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__(input_dim, hidden_dim, embedding_dim, num_heads, dropout)
        self.cell_type_attention = nn.Linear(embedding_dim + num_cell_types, 1)
        self.register_buffer("gene_interactions", gene_interactions)

        # MLP to process gene interaction features
        self.interaction_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(embedding_dim)
        )
    
    def compute_gene_interaction_features(self, x):
        """
        Compute cell features based on gene interactions.
        
        Args:
            x: [num_cells x num_genes] Gene expression matrix
            
        Returns:
            torch.Tensor: [num_cells x embedding_dim] Interaction-aware features
        """
        if self.gene_interactions is None:
            return torch.zeros(x.shape[0], self.embedding_dim, device=x.device)
            
        # Normalize gene expression
        x_norm = F.normalize(x, dim=1)
        
        # For each cell, compute interaction strength between its expressed genes
        # [num_cells x num_genes] @ [num_genes x num_genes] @ [num_genes x num_cells]
        interaction_scores = x_norm @ self.gene_interactions @ x_norm.T
        
        # Project interaction scores to embedding dimension
        interaction_features = self.interaction_encoder(interaction_scores)
        
        return interaction_features

    def aggregate_cell_type_embeddings(self, cell_embeddings, cell_types):
        """
        Aggregate embeddings for each cell type by taking the mean.
        
        Args:
            cell_embeddings: [num_cells x embedding_dim] Cell embeddings
            cell_types: [num_cells x num_cell_types] One-hot cell type labels
            
        Returns:
            torch.Tensor: [num_cell_types x embedding_dim] Cell type embeddings
        """
        num_cell_types = cell_types.shape[1]
        cell_type_embeddings = []
        
        for cell_type_idx in range(num_cell_types):
            # Get mask for current cell type
            type_mask = cell_types[:, cell_type_idx]
            
            if type_mask.sum() == 0:
                # If no cells of this type, use zeros
                cell_type_embeddings.append(
                    torch.zeros(cell_embeddings.shape[0], device=cell_embeddings.device)
                )
            else:
                # Get embeddings for this cell type and take mean
                type_embeddings = cell_embeddings[type_mask > 0].mean(dim=0)
                cell_type_embeddings.append(type_embeddings)
            
        return torch.stack(cell_type_embeddings)

    def forward(self, x, edge_index, cell_type):
        """Forward pass with gene interaction features"""
        # Get base cell embeddings
        cell_embeddings = self.encode(x, edge_index)
        
        # Compute interaction-aware features
        interaction_features = self.compute_gene_interaction_features(x)
        
        # Add interaction features to cell embeddings
        cell_embeddings = cell_embeddings + interaction_features
        
        cell_type_context = torch.cat([cell_embeddings, cell_type], dim=1)
        attention_weights = torch.sigmoid(self.cell_type_attention(cell_type_context))
        cell_embeddings = cell_embeddings * attention_weights
        
        # Aggregate cell type embeddings
        cell_type_embeddings = self.aggregate_cell_type_embeddings(cell_embeddings, cell_type)
        
        return cell_embeddings, cell_type_embeddings