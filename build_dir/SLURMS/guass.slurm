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
hw_path=~matthewt1441/ece569/Fusion_Image_Upscale_HPC/build_dir

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
SCALES[5]=10;

DIM_X[0]=1;
DIM_X[1]=2;
DIM_X[2]=4;
DIM_X[3]=8;
DIM_X[4]=16;
DIM_X[5]=32;
DIM_X[6]=64;
DIM_X[7]=128;
DIM_X[8]=256;
DIM_X[9]=512;
DIM_X[10]=1024;

DIM_Y[0]=1;
DIM_Y[1]=2;
DIM_Y[2]=4;
DIM_Y[3]=8;
DIM_Y[4]=16;
DIM_Y[5]=32;
DIM_Y[6]=64;
DIM_Y[7]=128;
DIM_Y[8]=256;
DIM_Y[9]=512;
DIM_Y[10]=1024;



for game in ${GAMES[@]}
do
   for scale in ${SCALES[@]}
    do
        # for block_dim_y in ${DIM_Y[@]}
        # do
        #     for block_dim_x in ${DIM_X[@]}
        #     do
                file=gaus_$((GAME_COUNT))_$((scale)).txt
                ./GUAS_TEST_BENCH_Solution $scale $game -t vector > Kernel_Testing/GAUSS/$file
        #     done
        # done
    done
    GAME_COUNT=$((GAME_COUNT+1))
done
