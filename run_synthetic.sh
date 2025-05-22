#!/bin/bash
source ~/.bashrc

# Setting A
for ratio in 0.1 0.3 0.5 0.7 0.9
do
  for seed in $(seq 0 80)
  do
    python experiments/synthetic_experiments.py --setting "A" --ratio $ratio --seed $seed
  done
done

# Setting B
for ratio in 0.2 0.3 0.4 0.5
do
  for seed in $(seq 0 80)
  do
    python experiments/synthetic_experiments.py --setting "B" --ratio $ratio --seed $seed
  done
done

# Setting C
for ratio in 0 0.2 0.4 0.6 0.8
do
  for seed in $(seq 0 80)
  do
    python experiments/synthetic_experiments.py --setting "C" --ratio $ratio --seed $seed
  done
done
