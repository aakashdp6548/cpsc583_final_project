import argparse
import torch
from torch.optim import Adam
from pathlib import Path
import numpy as np
from torch.nn import utils

from models.gnn import InteractionGNN
from models.loss import InteractionLoss
from visualization import plot_embeddings, plot_loss_curves, plot_expression_matrix, plot_spatial_coords
from data.graph import SpatialGraphConstructor
from data.utils import create_combined_dataset
from torch_geometric.data import Data


def parse_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--num_cells", type=int, default=200)
    parser.add_argument("--num_genes", type=int, default=500)
    parser.add_argument("--num_cell_types", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--spatial_weight", type=float, default=1.0)
    parser.add_argument("--interaction_weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create combined dataset
    combined_data = create_combined_dataset(
        num_cells=args.num_cells,
        num_genes=args.num_genes,
        num_cell_types=args.num_cell_types
    )

    # Create a identity gene interaction matrix
    gene_interaction_matrix = torch.eye(args.num_genes)

    # Plot initial spatial data
    initial_save_path = output_dir / "initial_spatial_data.png"
    plot_spatial_coords(combined_data["spatial_coords"], combined_data["cell_types"], save_path=initial_save_path)

    # Initialize graph constructor
    graph_constructor = SpatialGraphConstructor(k_neighbors=10)

    # Create single graph
    data = Data(
        x=combined_data["gene_expression"],
        edge_index=graph_constructor.build_graph(combined_data["spatial_coords"]),
        pos=combined_data["spatial_coords"],
        cell_type=combined_data["cell_types"],
    ).to(device)

    # Initialize model
    model = InteractionGNN(
        input_dim=args.num_genes,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_cell_types=args.num_cell_types,
        gene_interactions=gene_interaction_matrix  # Your gene interaction matrix
    ).to(device)

    # Initialize loss and optimizer
    criterion = InteractionLoss(
        temperature=args.temperature,
        spatial_weight=args.spatial_weight,
        interaction_weight=args.interaction_weight
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Initialize loss tracking
    losses = {}

    # Training loop
    for epoch in range(args.num_epochs):
        # Forward pass
        model.train()
        optimizer.zero_grad()
        
        embeddings, _ = model(data.x, data.edge_index, data.cell_type)
        
        # Compute loss
        loss, loss_dict = criterion(
            embeddings=embeddings,
            data=data,
            gene_interactions=model.gene_interactions
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        for key, val in loss_dict.items():
            if key not in losses:
                losses[key] = []
            losses[key].append(val)

        # Print progress
        print(f"Epoch {epoch}:")
        for key, val in loss_dict.items():
            print(f"  {key}: {val:.4f}")
        
        # Visualize embeddings periodically
        if epoch % 10 == 0:
            # Plot and save embeddings
            save_path = output_dir / f'embeddings_epoch_{epoch}.png'
            plot_embeddings(embeddings, data.cell_type, save_path=save_path)

        # Plot loss curves
        plot_loss_curves(losses, save_path=output_dir / 'loss_curves.png')

if __name__ == "__main__":
    main()