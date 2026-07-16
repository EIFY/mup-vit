#!/bin/bash

scripts=("corrected_lr.sh" "corrected_wd.sh" "corrected_nesterov.sh" "corrected_momentum.sh" "corrected_sign_lr.sh" "corrected_sign_wd.sh" "corrected_lr_eff_transfer.sh" "corrected_mo_baseline_comparison.sh" "corrected_bias.sh" "corrected_c_sq_lr.sh" "corrected_cos_training_budgets.sh" "corrected_power.sh" "corrected_cos_power.sh" "corrected_cosine_power_comparison.sh" "corrected_training_budgets.sh" "lr.sh" "wd.sh" "nesterov.sh" "momentum.sh" "sign_lr.sh" "sign_wd.sh" "cos_training_budgets.sh" "power.sh" "cos_power.sh" "cosine_power_comparison.sh" "training_budgets.sh" "done")

len=${#scripts[@]}

python script_gen.py

while [ ! -f done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python script_gen.py
			break
		fi
	done
done

