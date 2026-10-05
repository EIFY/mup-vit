#!/bin/bash

scripts=("lr.sh" "training_budgets.sh" "done")
scripts=("${scripts[@]/#/corrected_long_}")

len=${#scripts[@]}

python corrected_long_script_gen.py

while [ ! -f corrected_long_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python corrected_long_script_gen.py
			break
		fi
	done
done
