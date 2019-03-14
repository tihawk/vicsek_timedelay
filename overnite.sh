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
folder="sortme/delay"
mkdir $folder
cd $folder/

# === Manually set up overnight simulations ===

for L in 4.25 3.75 4.75
do
	echo 32 $L Static number $i
	python $HOME/__main__.py 32 $L 1 1 1
	wait
done

#for L in 6.5 7. 7.5 8. 8.5 9. 9.5 10. 10.5 11. 11.5 12.
#do
#	echo 64 $L Static number $i
#	python $HOME/__main__.py 64 $L 0 1 1
#	wait
#done

#for L in 4.75 5.25
#do
#	echo 128 $L Static number $i
#	python $HOME/__main__.py 128 $L 0 1 1
#	wait
#done

#for L in 5.75
#do
#	echo 256 $L Static number $i
#	python $HOME/__main__.py 256 $L 0 1 1
#	wait
#done

#for L in 7.5 8. 8.5 5. 4.5 4.
#do
#	echo 512 $L Static number $i
#	python $HOME/__main__.py 512 $L 0 1 1
#	wait
#done
#
#for L in 7.5 7. 6.5
#do
#	echo 1024 $L Static number $i
#	python $HOME/__main__.py 1024 $L 0 1 1
#	wait
#done

# ============================================

read -p "Press Enter to close..."