#!/bin/bash
source ~/.bashrc

for folder in {1..77}
do
  for k in {0..4}
  do
    python experiments/ACIC_experiments.py --folder $folder --k $k
  done
done
