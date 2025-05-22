#!/bin/bash
source ~/.bashrc

for i in {0..999}; do
    python experiments/IHDP_experiments.py --i $i
done
