import argparse
import torch
from torch.optim import Adam
from pathlib import Path
import numpy as np
from torch.nn import utils
import json
import random

from models.gnn import InteractionGNN
from models.loss import InteractionLoss
from visualization import plot_embeddings, plot_loss_curves, plot_expression_matrix, plot_spatial_coords
from data.graph import SpatialGraphConstructor
from data.utils import create_combined_dataset, load_data
from torch_geometric.data import Data


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    # Debugging arguments
    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed right after parsing args
    set_seed(args.seed)
    
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
    graph_constructor = SpatialGraphConstructor(k_neighbors=10, seed=args.seed)

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

    # Initialize loss tracking and early stopping variables
    losses = {}
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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
        
        # Track losses
        for key, val in loss_dict.items():
            if key not in losses:
                losses[key] = []
            losses[key].append(val)

        # Print progress
        print(f"Epoch {epoch}:")
        for key, val in loss_dict.items():
            print(f"  {key}: {val:.4f}")
        
        # Early stopping check
        current_loss = loss_dict['total_loss']
        if current_loss < best_loss - args.min_delta:
            best_loss = current_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                # Load best model state
                model.load_state_dict(best_model_state['model_state_dict'])
                break
        
        # Visualize embeddings periodically
        if epoch % plot_every == 0:
            save_path = output_dir / f'embeddings_epoch_{epoch}.png'
            title = f"{args.dataset} Embeddings - Epoch {epoch}"
            plot_embeddings(embeddings, data.cell_type, save_path=save_path, title=title)

    # Plot loss curves
    plot_loss_curves(losses, save_path=output_dir / 'loss_curves.png')

    # Get final embeddings using best model
    model.eval()
    with torch.no_grad():
        final_embeddings, _ = model(data.x, data.edge_index, data.cell_type)

    # Plot and save final embeddings
    save_path = output_dir / f'embeddings_final.png'
    title = f"{args.dataset} Embeddings - Final"
    plot_embeddings(final_embeddings, data.cell_type, save_path=save_path, title=title)

    # Save final embeddings
    final_embeddings_path = output_dir / "final_embeddings.pt"
    torch.save(final_embeddings.detach().cpu(), final_embeddings_path)

    # Save best model state
    model_save_path = output_dir / "best_model.pt"
    torch.save(best_model_state, model_save_path)

    # Save losses
    losses_path = output_dir / "losses.json"
    with open(losses_path, "w") as f:
        json.dump(losses, f, indent=4)

if __name__ == "__main__":
    main()