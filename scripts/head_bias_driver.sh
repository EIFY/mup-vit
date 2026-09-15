#!/bin/bash

scripts=("corrected_head_bias_wd.sh" "corrected_head_bias_training_budgets.sh" "head_bias_wd.sh" "head_bias_training_budgets.sh" "head_bias_done")

len=${#scripts[@]}

python head_bias_script_gen.py

while [ ! -f head_bias_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python head_bias_script_gen.py
			break
		fi
	done
done

