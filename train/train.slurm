#!/bin/bash

#SBATCH --job-name=test_lstm
#SBATCH --output=test_lstm.out

# Use the gpu partition and request one GPU
#SBATCH --partition=gpu

# Request memory
#SBATCH --mem=16G

# Set maximum run time
#SBATCH --time=1-12:00:00

# Node list
#SBATCH --nodes=1
#SBATCH --exclude=node[214-217]

# Activate conda environment
module load soft/anaconda3/2020.07/latest
ENV="tensorflow2-gpu"
source activate $ENV
cd /m/scratch/hive/alim/NLP/SentimentTinker/train

# Run script
python lstm_bayes.py /m/scratch/hive/alim/NLP/SentimentTinker/data/train_valid_test_data.pkl 10000
