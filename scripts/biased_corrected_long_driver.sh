#!/bin/bash

scripts=("lr.sh" "c_sq_lr.sh" "training_budgets.sh" "done")
scripts=("${scripts[@]/#/biased_corrected_long_}")

len=${#scripts[@]}

python biased_corrected_long_script_gen.py

while [ ! -f biased_corrected_long_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python biased_corrected_long_script_gen.py
			break
		fi
	done
done
