# Predicting cell-cell communication with spatial transcriptomics and interaction priors

This repository contains the code for the project "Modeling Interaction Priors in GNNs for Predicting Cell-Cell Communication".

## Project Description

This project aims to predict cell-cell communication using spatial transcriptomics data and known ligand-receptor interactions. We propose a novel approach that incorporates interaction priors into graph neural networks (GNNs) within a self-supervised learning framework. We show that this approach outperforms methods that rely only on gene expression and spatial proximity for predicting cell-cell communication.

## Repository Structure

The repository is organized as follows:

- `src/data/`: Contains the code for loading and preprocessing the data.
- `src/models/`: Contains the code for the models used in the project.
- `src/visualization.py`: Contains the code for visualizing the results including UMAPs of embeddings and plotting loss curves.
- `train.py`: Main training script. Runs the training loop and saves the embeddings.


## Usage

Install the dependencies using `pip install -r requirements.txt`. For [CellPhoneDB](https://github.com/ventolab/CellphoneDB-data) and [SpaceFlow](https://github.com/hongleir/SpaceFlow/tree/master), please refer to the documentation for how to download and install the software.  The code assumes that the datasets are located in the `data/` directory.

The data is too large to upload to the repository. Instead, run the `download_and_process_data.ipynb` notebook to download the datasets. To construct the interaction priors, run the `get_cpdb_interactions.ipynb` notebook.

The training script can be run as follows:
```python
python train.py \
    --dataset <dataset_name> \  # lymph_node or breast_cancer
    --output_dir <output_dir> \ # directory to save the embeddings
    --num_genes <num_genes> \ # number of genes to use
    --hidden_dim <hidden_dim> \ # hidden dimension of the GNN
    --embedding_dim <embedding_dim> \ # embedding dimension
    --num_heads <num_heads> \ # number of attention heads
    --dropout <dropout> \ # dropout rate
    --learning_rate <learning_rate> \ # learning rate
    --num_epochs <num_epochs> \ # number of epochs
    --temperature <temperature> \ # temperature for contrastive loss
    --contrastive_weight <contrastive_weight> \ # weight for contrastive loss
    --spatial_weight <spatial_weight> \ # weight for spatial loss
    --interaction_weight <interaction_weight> \ # weight for interaction loss
    --patience <patience> \ # patience for early stopping
    --min_delta <min_delta> \ # minimum delta for early stopping
    --plot_every <plot_every> \ # plot every x epochs
    --seed <seed> \ # random seed
```

Alternatively, run the `run_model.sh` script to train the model on each dataset with three random seeds.

To make final predictions and get evaluation results, run the `predict_interactions.ipynb` notebook.

To evaluate SpaceFlow, run the `spaceflow.ipynb` notebook.