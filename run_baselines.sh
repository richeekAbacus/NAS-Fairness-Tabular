#!/bin/bash
# Run all the baselines

dataset=${1:-"adult"}
protected=${2:-"sex"}
log_name="logs/baselines_${dataset}_${protected}.log"

echo "Running baselines for $dataset with protected attribute $protected"
echo "Logging to $log_name"
rm $log_name


methods=("reweighing" "disparate_impact_remover"  "lfr" "optim_proc"
         "adversarial_debiasing" "gerryfair" "metafair" "prejudice_remover"
         "exponentiated_gradient_reduction" "grid_search_reduction"
         "calibrated_eq_odds" "eq_odds" "reject_option_classification")

for method in "${methods[@]}"
do
    echo "Running $method"
    python3 baselines.py --dataset $dataset --privilege_mode $protected --debiaser $method --log $log_name
done
