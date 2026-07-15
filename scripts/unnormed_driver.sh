#!/bin/bash

scripts=("unnormed_corrected_lr.sh" "unnormed_corrected_wd.sh" "unnormed_corrected_sign_lr.sh" "unnormed_corrected_sign_wd.sh" "unnormed_corrected_c_sq_lr.sh" "unnormed_wd.sh" "done")

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
			python unnormed_script_gen.py
			break
		fi
	done
done

