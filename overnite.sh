#!/usr/bin/bash

# =================================================
# This script is for the purposes of setting up
# a custom set of simulations to be ran during AFK.
# For a more user-friendly automated script, check
# "run.sh".
# =================================================

# directory with scripts
HOME=/D/DevelopiNation/thesis/vicsek_timedelay

# index of the variables
#N=$1
#L=$2
#dt=$3
#isStatic=$4
#k=$5

# make directory for current simulation
# NOTE: puts everything in a "HOME/sortme" folder
folder="sortme"
mkdir $folder
cd $folder/

# === Manually set up overnight simulations ===

for L in 11.5 12. 12.5 13.
do
	for i in 1 2 3 4 5
	do
		echo 1024 $L Static number i
		python $HOME/__main__.py 1024 $L 0 "True" 1
		wait
	done
done

for L in 10. 10.5 9.
do
	for i in 1 2 3 4 5
	do
		echo 512 $L Static number i
		python $HOME/__main__.py 512 $L 0 "True" 1
		wait
	done
done

for L in 8.25 8.75 9.
do
	for i in 1 2 3 4 5
	do
		echo 256 $L Static number i
		python $HOME/__main__.py 256 $L 0 "True" 1
		wait
	done
done

# ============================================

read -p "Press Enter to close..."