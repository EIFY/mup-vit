#!/bin/bash

scripts=("ascion_alpha.sh" "ascion_lr_eff_transfer.sh" "ascion_done")
len=${#scripts[@]}

python ascion_script_gen.py

while [ ! -f ascion_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python ascion_script_gen.py
			break
		fi
	done
done
