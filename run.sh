#!/usr/bin/bash

# directory with scripts
HOME=/D/DevelopiNation/thesis/vicsek_timedelay

# variables for running the simulations
N=$1
L=$2
dt=$3
isStatic=$4
k=$5

echo "N =" $1
echo "L =" $2
echo "time delay =" $3
echo "static correlation =" $4
echo "k for spatio-temporal correlation =" $5

# according to the variables, sets up where
# the data gets saved
case $isStatic in
	1)
	
		case $dt in
			0)
				folder="static_corr_nodelay/new/"$N
				;;
			*)
				folder="static_corr_delay/new/"$N
				;;
		esac
		;;
		
	0)
	
		case $dt in
			0)
				folder="spattemp_corr_nodelay/"
				;;
			*)
				folder="spattemp_corr_delay/"
				;;
		esac
		;;
				
	*)
		echo "Input valid variables while starting the script!!!"
		;;
esac

# make directory for current simulation
mkdir $folder
cd $folder/
echo "Saving data in:"
pwd

# run vicsek simulation
# runs a set of 5 simulations for static correlation function calc
# runs 1 simulation for spatio-temporal correlation function calc
case $isStatic in
	1)
		for i in 1 2 3 4 5
		do
			echo "Running 5 simulations consecutively"
			echo "Starting number $i/5"
			python $HOME/__main__.py $N $L $dt $isStatic $k
			wait
		done
		;;
	0)
		echo "Running one simulation"
		python $HOME/__main__.py $N $L $dt $isStatic $k
		wait
		;;
esac

# don't close terminal on finish
read -p "Press Enter to close..."