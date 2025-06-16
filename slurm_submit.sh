#!/bin/bash

slurm_script="slurm_ihdp.sh"

for i in {0..999}; do
    sbatch $slurm_script $i
done

# slurm_script="slurm_acic.sh"

# for folder in {1..77}
# do
#   for k in {0..4}
#   do
#     sbatch $slurm_script $folder $k
#   done
# done

# slurm_script="slurm_synthetic.sh"

# for ratio in 0.1 0.3 0.5 0.7 0.9
# do
#   for seed in $(seq 0 80)
#   do
#     sbatch $slurm_script "A" $ratio $seed
#   done
# done

# for ratio in 0.2 0.3 0.4 0.5
# do
#   for seed in $(seq 0 80)
#   do
#     sbatch $slurm_script "B" $ratio $seed
#   done
# done

# for ratio in 0 0.2 0.4 0.6 0.8
# do
#   for seed in $(seq 0 80)
#   do
#     sbatch $slurm_script "C" $ratio $seed
#   done
# done