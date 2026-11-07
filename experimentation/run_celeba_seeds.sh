#!/bin/zsh
# Semillas 1 y 2 de CelebA desde training.ipynb: por cada semilla, las 3 variantes en paralelo; una semilla tras otra.
cd "$(dirname "$0")"
for seed in 1 2; do
  for name in unconstrained constrained constrained_rank; do
    ../venv/bin/python run_training_notebook.py --name $name --seed $seed > logs/nbrun_${name}_s${seed}.log 2>&1 &
  done
  wait
done
echo ALL_DONE > logs/nbrun_ALL_DONE
