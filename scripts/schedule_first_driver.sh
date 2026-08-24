#!/bin/bash

scripts=("cos_power.sh" "power.sh" "cosine_power_comparison.sh" "lr.sh" "wd.sh" "nesterov.sh" "momentum.sh" "sign_lr.sh" "sign_wd.sh" "bias.sh" "c_sq_lr.sh" "training_budgets.sh" "done")
scripts=("${scripts[@]/#/schedule_first_}")

len=${#scripts[@]}

python schedule_first_script_gen.py

while [ ! -f done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python schedule_first_script_gen.py
			break
		fi
	done
done

