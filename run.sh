#!/bin/bash

#SBATCH -n 10                         # number of cores
#SBATCH --time=23:59:59               # time limit on the cluster (days-hours:minutes:seconds)
#SBATCH --mem-per-cpu=12000           # memory per processor core (MB)
#SBATCH --tmp=12000                   # temporary data space used during the job (MB)
#SBATCH --job-name=analysis1
#SBATCH --output=analysis1.out        # output file
#SBATCH --error=analysis1.err         # error output file

module purge
module load StdEnv
module load python

python runCollectiveDisplacement.py
python plots.py