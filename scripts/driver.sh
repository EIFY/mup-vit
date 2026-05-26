#!/bin/bash

scripts=("corrected_lr.sh" "corrected_wd.sh" "corrected_nesterov.sh" "corrected_momentum.sh" "corrected_sign_lr.sh" "corrected_sign_wd.sh" "corrected_lr_eff_transfer.sh" "corrected_training_budgets.sh" "corrected_log_time_momentum.sh" "corrected_baseline_comparison.sh" "corrected_log_time_training_budgets.sh" "lr.sh" "wd.sh" "nesterov.sh" "momentum.sh" "sign_lr.sh" "sign_wd.sh" "training_budgets.sh" "misc.sh" "done")

len=${#scripts[@]}

python script_gen.py

for ((i=0; i<$len-1; i++)); do
	curr="${scripts[i]}"
	next="${scripts[i+1]}"
	while [ ! -f $next ]; do
		echo "$curr -> $next"
		bash $curr
		python script_gen.py
	done
done