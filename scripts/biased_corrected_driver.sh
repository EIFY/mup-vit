#!/bin/bash

scripts=("lr.sh" "wd.sh" "nesterov.sh" "momentum.sh" "sign_lr.sh" "sign_wd.sh" "bias.sh" "c_sq_lr.sh" "power.sh" "cos_power.sh" "cosine_power_comparison.sh" "training_budgets.sh" "done")
scripts=("${scripts[@]/#/biased_corrected_}")

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

