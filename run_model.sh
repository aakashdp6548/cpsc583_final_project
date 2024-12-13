#!/bin/bash

NUM_GENES=2000
LEARNING_RATE=0.0001
NUM_EPOCHS=200
TEMPERATURE=0.7
SPATIAL_WEIGHT=1.0
INTERACTION_WEIGHT=10.0
PLOT_EVERY=100

for dataset in lymph_node breast_cancer
do
    for seed in 13 21 42
    do
        python train.py \
        --dataset $dataset \
        --output_dir training_output/${dataset}_seed_${seed} \
        --num_genes $NUM_GENES \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --temperature $TEMPERATURE \
        --spatial_weight $SPATIAL_WEIGHT \
        --interaction_weight $INTERACTION_WEIGHT \
        --plot_every $PLOT_EVERY \
        --seed $seed
    done
done
