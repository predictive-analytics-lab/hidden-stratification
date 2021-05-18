#!/bin/bash
for seed in 3 4 5 6 7
do
        sbatch celeba.job data_split_seed=$seed
done
