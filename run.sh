#!/bin/bash
# Run the NAS-Fairness-Tabular experiment

fairness_metric=${1:-"statistical_parity_difference"}
run_name=${2:-"statistical_parity_difference"}
output_dir=${3:-"results/"}

python3 train.py \
    --dataset adult \
    --train_bs 64 \
    --test_bs 64 \
    --model FTTransformer \
    --multi_objective \
    --fairness_metric $fairness_metric \
    --run_name $run_name \
    --output_dir $output_dir \
    --wall_time_limit 1000000 \
    --n_trials 200 \
    --initial_n_configs 10 \
    --min_budget 2.5 \
    --max_budget 10 \
    --eval_budget 10 \
    --successive_halving_eta 3 \
    --seed 42 \
    &> logs/$run_name.log
