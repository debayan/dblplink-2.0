#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1

python candidate_reranker.py
