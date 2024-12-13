import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractionLoss(nn.Module):
    def __init__(self, temperature=0.1, contrastive_weight=1.0, spatial_weight=1.0, interaction_weight=1.0):
        """
        Loss function combining cell type contrastive loss with spatial regularization.
        
        Args:
            temperature (float): Temperature parameter for similarity scaling
            spatial_weight (float): Weight for spatial regularization term
            interaction_weight (float): Weight for interaction regularization term
        """
        super().__init__()
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.spatial_weight = spatial_weight
        self.interaction_weight = interaction_weight
    
    def compute_spatial_loss(self, embeddings, spatial_coords):
        """
        Compute loss term that encourages spatially close cells to have similar embeddings.
        
        Args:
            embeddings: [N x D] Cell embeddings
            spatial_coords: [N x 2] Spatial coordinates
            
        Returns:
            torch.Tensor: Spatial regularization loss
        """
        # Compute pairwise distances in euclidean space
        spatial_dists = torch.cdist(spatial_coords, spatial_coords)
        spatial_dists = spatial_dists / spatial_dists.max()  # normalize to [0, 1]
        spatial_sims = 1 - spatial_dists  # closer = more similar
        
        # Compute pairwise similarities in embedding space
        embeddings = F.normalize(embeddings, dim=1)
        embedding_sims = embeddings @ embeddings.T
        
        # Mask out self-similarities
        mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        spatial_sims = spatial_sims * (1 - mask)
        embedding_sims = embedding_sims * (1 - mask)

        # Compute loss as mean squared difference between similarity matrices
        spatial_loss = F.mse_loss(embedding_sims, spatial_sims)
        return spatial_loss
    

    def compute_cell_type_loss(self, embeddings, cell_types):
        """
        Compute cell type contrastive loss.

        Args:
            embeddings: [N x D] Cell embeddings
            cell_types: [N x num_cell_types] One-hot cell type labels

        Returns:
            torch.Tensor: Cell type contrastive loss
        """
        # Compute pairwise similarities in embedding space
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = (embeddings @ embeddings.T) / self.temperature

        # Create label matrix where (i,j) = 1 if same cell type, 0 otherwise
        cell_type_indices = torch.argmax(cell_types, dim=1)
        labels = (cell_type_indices.unsqueeze(0) == cell_type_indices.unsqueeze(1)).float()

        # Remove self-connections from positive pairs
        mask = torch.eye(labels.shape[0], device=labels.device)
        labels = labels * (1 - mask)

        # Compute contrastive loss
        log_probs = F.log_softmax(sim_matrix, dim=1)
        contrastive_loss = -(log_probs * labels).sum(dim=1) / labels.sum(dim=1).clamp(min=1)
        contrastive_loss = contrastive_loss.mean()
        return contrastive_loss


    def compute_interaction_loss(self, embeddings, gene_expression, gene_interactions):
        """
        Compute loss term that encourages embeddings to reflect gene interactions.
        
        Args:
            embeddings: [N x D] Cell embeddings
            gene_expression: [N x G] Gene expression matrix
            gene_interactions: [G x G] Gene interaction matrix
            
        Returns:
            torch.Tensor: Interaction loss
        """
        # Normalize gene expression
        gene_expression = F.normalize(gene_expression, dim=1)
        
        # Compute expected interaction strength between cells
        # based on their gene expression and known gene interactions
        expected_interactions = gene_expression @ gene_interactions @ gene_expression.T
        expected_interactions = expected_interactions / expected_interactions.max()
        
        # Compute actual similarities in embedding space
        embeddings = F.normalize(embeddings, dim=1)
        embedding_sims = embeddings @ embeddings.T
        
        # Mask out self-interactions
        mask = torch.eye(embeddings.shape[0], device=embeddings.device)
        expected_interactions = expected_interactions * (1 - mask)
        embedding_sims = embedding_sims * (1 - mask)
        
        # Compute loss
        interaction_loss = F.mse_loss(embedding_sims, expected_interactions)
        
        return interaction_loss


    def forward(self, embeddings, data, gene_interactions):
        """
        Compute total loss combining cell type contrastive loss, spatial regularization, and interaction loss.
        
        Args:
            embeddings: [N x D] Cell embeddings
            data: PyG Data object containing cell types, spatial coordinates, and gene expression
            gene_interactions: [G x G] Gene interaction matrix
            
        Returns:
            torch.Tensor: Combined loss
        """
        cell_types = data.cell_type
        spatial_coords = data.pos
        gene_expression = data.x

        contrastive_loss = self.compute_cell_type_loss(embeddings, cell_types)
        spatial_loss = self.compute_spatial_loss(embeddings, spatial_coords)
        interaction_loss = self.compute_interaction_loss(embeddings, gene_expression, gene_interactions)
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.spatial_weight * spatial_loss +
            self.interaction_weight * interaction_loss
        )

        loss_dict = {
            'contrastive_loss': self.contrastive_weight * contrastive_loss.item(),
            'spatial_loss': self.spatial_weight * spatial_loss.item(),
            'interaction_loss': self.interaction_weight * interaction_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
