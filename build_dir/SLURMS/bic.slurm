#!/bin/bash

### Set the job name 
#SBATCH --job-name=ece569hw4

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:03:00

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
hw_path=~kylerm23/ece569/build_dir

cd $hw_path


GAMES[0]=./NV/image;
GAMES[1]=./LAD/LAD_;
GAMES[2]=./LM_Frame/image;

GAME_COUNT=0;

SCALES[0]=2;
SCALES[1]=3;
SCALES[2]=4;
SCALES[3]=6;
SCALES[4]=8;

DIM[0]=8;
DIM[1]=16;
DIM[2]=32;

COUNTS[0]=0;
COUNTS[1]=1;
COUNTS[2]=2;
COUNTS[3]=3;
COUNTS[4]=4;


for game in ${GAMES[@]}
do
   for scale in ${SCALES[@]}
    do
        for block_dim in ${DIM[@]}
        do
            for count in ${COUNTS[@]}
            do
                file=bic_$((GAME_COUNT))_$((scale))_$((block_dim))_$((count)).txt
                ./Histogram_Solution $scale $block_dim $game -t vector > Kernel_Testing/$file
            done
        done
    done
    GAME_COUNT=$((GAME_COUNT+1))
done