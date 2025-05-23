#!/bin/bash
#SBATCH --array 0-4
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=128G
#SBATCH --time=24:00:00

python main.py ${SLURM_ARRAY_TASK_ID}