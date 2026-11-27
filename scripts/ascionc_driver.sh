#!/bin/bash

scripts=("ascionc_alpha.sh" "ascionc_lr_eff_transfer.sh" "ascionc_done")
len=${#scripts[@]}

python ascionc_script_gen.py

while [ ! -f ascionc_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python ascionc_script_gen.py
			break
		fi
	done
done
