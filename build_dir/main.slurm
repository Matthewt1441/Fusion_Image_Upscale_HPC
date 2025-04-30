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


GAMES[0]=./NV/images;
GAMES[1]=./LAD/images;
GAMES[2]=./LM/images;

GAMENAMES[0]="FALLOUT_NEW_VEGAS";
GAMENAMES[1]="LIKE_A_DRAGON_INFINITE_WEALTH";
GAMENAMES[2]="LUIGIS_MANSION";

SCALES[0]=2;
SCALES[1]=3;
SCALES[2]=4;
SCALES[3]=6;
SCALES[4]=8;

start_frame=1;
end_frame=30;

LOOPS[0]=1;
LOOPS[1]=2;
LOOPS[2]=3;

for scale in ${SCALES[@]}
do
  for loop in ${LOOPS[@]}
  do
    #Stream Count = 1
    file=cuda_code_NV_$((scale))__$((loop)).txt
    ./CUDA_TEST_BENCH_Solution $scale ./NV/images "Fallout New Vegas" $start_frame $end_frame -t vector > Kernel_Testing/$file

    file=cuda_code_LAD_$((scale))__$((loop)).txt
    ./CUDA_TEST_BENCH_Solution $scale ./LAD/images "LIKE A DRAGON INFINITE WEALTH" $start_frame $end_frame -t vector > Kernel_Testing/$file

    file=cuda_code_LM_$((scale))__$((loop)).txt
    ./CUDA_TEST_BENCH_Solution $scale ./LM/images "LUIGIS MANSION" $start_frame $end_frame -t vector > Kernel_Testing/$file


    for stream in ${STREAMS[@]}
    do
      file=cuda_stream_$((stream))_code_NV_$((scale))__$((loop)).txt
      ./CUDA_STREAM_TEST_BENCH_Solution $scale ./NV/images "Fallout New Vegas" $start_frame $end_frame $stream -t vector > Kernel_Testing/$file

      file=cuda_stream_$((stream))_code_LAD_$((scale))__$((loop)).txt
      ./CUDA_STREAM_TEST_BENCH_Solution $scale ./LAD/images "LIKE A DRAGON INFINITE WEALTH" $start_frame $end_frame $stream -t vector > Kernel_Testing/$file

      file=cuda_stream_$((stream))_code_LM_$((scale))__$((loop)).txt
      ./CUDA_STREAM_TEST_BENCH_Solution $scale ./LM/images "LUIGIS MANSION" $start_frame $end_frame $stream -t vector > Kernel_Testing/$file
    done
  done
done
