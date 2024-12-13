import argparse
import torch
from torch.optim import Adam
from pathlib import Path
import numpy as np
from torch.nn import utils
import json

from models.gnn import InteractionGNN
from models.loss import InteractionLoss
from visualization import plot_embeddings, plot_loss_curves, plot_expression_matrix, plot_spatial_coords
from data.graph import SpatialGraphConstructor
from data.utils import create_combined_dataset, load_data
from torch_geometric.data import Data


def parse_args():
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--dataset", type=str, default="lymph_node")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_genes", type=int, default=2000)

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

    # Debugging arguments
    parser.add_argument("--plot_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save training args to output_dir as training_args.json
    args_dict = vars(args)
    args_save_path = output_dir / "training_args.json"
    with open(args_save_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    
    # Load dataset and interaction matrix
    data = load_data(args.dataset, args.num_genes)
    gene_interaction_matrix = data["interaction_matrix"]
    num_cell_types = data["cell_types"].shape[1]

    # Plot initial spatial data
    initial_save_path = output_dir / "initial_spatial_data.png"
    plot_spatial_coords(data["spatial_coords"], data["cell_types"], save_path=initial_save_path)

    # Initialize graph constructor
    graph_constructor = SpatialGraphConstructor(k_neighbors=10)

    # Create single graph
    data = Data(
        x=data["gene_expression"],
        edge_index=graph_constructor.build_graph(data["spatial_coords"]),
        pos=data["spatial_coords"],
        cell_type=data["cell_types"],
    ).to(device)

    # Initialize model
    model = InteractionGNN(
        input_dim=args.num_genes,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_cell_types=num_cell_types,
        gene_interactions=gene_interaction_matrix
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

    plot_every = args.plot_every if args.plot_every > 0 else args.num_epochs + 1

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
        if epoch % plot_every == 0:
            # Plot and save embeddings
            save_path = output_dir / f'embeddings_epoch_{epoch}.png'
            plot_embeddings(embeddings, data.cell_type, save_path=save_path, epoch=epoch)

        # Plot loss curves
        plot_loss_curves(losses, save_path=output_dir / 'loss_curves.png')

    # Plot and save embeddings at end of training
    save_path = output_dir / f'final_embeddings.png'
    plot_embeddings(embeddings, data.cell_type, save_path=save_path)

    # Save final embeddings
    final_embeddings_path = output_dir / "final_embeddings.pt"
    torch.save(embeddings, final_embeddings_path)

    # Save losses
    losses_path = output_dir / "losses.json"
    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=4)

if __name__ == "__main__":
    main()