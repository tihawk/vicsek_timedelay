#!/usr/bin/bash

# =================================================
# This script is for the purposes of setting up
# a custom set of simulations to be ran during AFK.
# For a more user-friendly automated script, check
# "run.sh".
# =================================================

# directory with scripts
HOME="$(cd "$(dirname "$0")"; pwd)"

# index of the variables
#N=$1
#L=$2
#dt=$3

# make directory for current simulation
# NOTE: change the folder to save to manually!
folder="particles/1delay"
#folder="sortme/0.5delay"
mkdir $folder
cd $folder/

# === Manually set up overnight simulations ===

#for L in 2.5 2.6 2.7 2.8 2.9 3. 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4. 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5. 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.
#do
#	echo 32 $L
#	python $HOME/__main__.py 32 $L 1
#	wait
#done

#for L in 3.5 3.6 3.7 3.8 3.9 4. 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5. 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6. 6.1 6.2 6.3 6.4 6.5
#do
#	echo 64 $L
#	python $HOME/__main__.py 64 $L 1
#	wait
#done

#for L in 5.8 5.9 6. 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7.
#do
#	echo 128 $L
#	python $HOME/__main__.py 128 $L 1
#	wait
#done
#
#for L in 6. 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7. 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9 8.
#do
#	echo 256 $L
#	python $HOME/__main__.py 256 $L 1
#	wait
#done
#
#for L in 7.5 7.6 7.7 7.8 7.9 8. 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9. 9.1 9.2 9.3 9.4 9.5
#do
#	echo 512 $L
#	python $HOME/__main__.py 512 $L 1
#	wait
#done

for L in 9.5 9.6 9.7 9.8 9.9 10. 10.1 10.2
do
	echo 1024 $L
	python $HOME/__main__.py 1024 $L 1
	wait
done

# ============================================

read -p "Press Enter to close..."