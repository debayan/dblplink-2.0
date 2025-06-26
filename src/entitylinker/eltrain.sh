#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1

python entity_linker.py
