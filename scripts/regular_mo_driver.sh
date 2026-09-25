#!/bin/bash

scripts=("corrected_regular_mo_training_budgets.sh" "regular_mo_training_budgets.sh" "regular_mo_done")

len=${#scripts[@]}

python head_bias_script_gen.py

while [ ! -f regular_mo_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python regular_mo_script_gen.py
			break
		fi
	done
done

