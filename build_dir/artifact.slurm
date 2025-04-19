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

scale=2
block_dim=8

file=artifact__$((scale))_$((block_dim))_1.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_2.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_3.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_4.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_5.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

block_dim=16

file=artifact__$((scale))_$((block_dim))_1.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_2.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_3.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_4.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_5.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

scale=4
block_dim=8

file=artifact__$((scale))_$((block_dim))_1.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_2.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_3.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_4.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_5.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

block_dim=16

file=artifact__$((scale))_$((block_dim))_1.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_2.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_3.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_4.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file

file=artifact__$((scale))_$((block_dim))_5.txt
./Histogram_Solution $scale $block_dim -t vector > Kernel_Testing/$file