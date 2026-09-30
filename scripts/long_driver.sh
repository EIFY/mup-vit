#!/bin/bash

scripts=("long_bias_wd.sh" "long_wd.sh" "long_done")

len=${#scripts[@]}

python long_script_gen.py

while [ ! -f long_done ]; do
	for ((i=0; i<$len-1; i++)); do
		curr="${scripts[i]}"
		next="${scripts[i+1]}"
		if [ ! -f $next ]; then
			echo "$curr -> $next"
			bash $curr
			rm ${scripts[*]}
			python long_script_gen.py
			break
		fi
	done
done

