#!/usr/bin/bash

# directory with scripts
HOME="$(cd "$(dirname "$0")"; pwd)"

# variables for running the simulations
N=$1
L=$2
dt=$3

echo "N =" $1
echo "L =" $2
echo "time delay =" $3

# make directory for current simulation
folder="particles/"$dt"delay/"
mkdir $folder
cd $folder/
echo "Saving data in:"
pwd

# run vicsek simulation
# runs a set of 5 simulations for static correlation function calc
# runs 1 simulation for spatio-temporal correlation function calc
echo "Starting..."
python $HOME/__main__.py $N $L $dt $isStatic $k
wait

# don't close terminal on finish
read -p "Press Enter to close..."