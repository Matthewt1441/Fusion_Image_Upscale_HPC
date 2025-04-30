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

SCALES[0]=2;
SCALES[1]=3;
SCALES[2]=4;
SCALES[3]=6;
SCALES[4]=8;

start_frame=1;
end_frame=15;


for scale in ${SCALES[@]}
do

  # Serial Code
  file=serial_code_LM_$((scale)).txt
  ./Serial_Fusion_Solution $scale ./LM/images "LM" $start_frame $end_frame -t vector > MAIN_OUTPUT/SERIAL_OUTPUT/$file
  
  file=serial_code_LAD_$((scale)).txt
  ./Serial_Fusion_Solution $scale ./LAD/images "LAD" $start_frame $end_frame -t vector > MAIN_OUTPUT/SERIAL_OUTPUT/$file
  
  file=serial_code_NV_$((scale)).txt
  ./Serial_Fusion_Solution $scale ./NV/images "NV" $start_frame $end_frame -t vector > MAIN_OUTPUT/SERIAL_OUTPUT/$file

  # # Optimized Code
  # file=optimized_code_LM_$((scale)).txt
  # ./Optimized_Cuda_Solution $scale ./LM/images "LM" $start_frame $end_frame -t vector > MAIN_OUTPUT/OPTIMIZED_OUTPUT/$file
  
  # file=optimized_code_LAD_$((scale)).txt
  # ./Optimized_Cuda_Solution $scale ./LAD/images "LAD" $start_frame $end_frame -t vector > MAIN_OUTPUT/OPTIMIZED_OUTPUT/$file
  
  # file=optimized_code_NV_$((scale)).txt
  # ./Optimized_Cuda_Solution $scale ./NV/images "NV" $start_frame $end_frame -t vector > MAIN_OUTPUT/OPTIMIZED_OUTPUT/$file

  # # Streamed Code
  # file=streamed_code_LM_$((scale)).txt
  # ./Streamed_Cuda_Solution $scale ./LM/images "LM" $start_frame $end_frame 3 -t vector > MAIN_OUTPUT/STREAM_OUTPUT/$file
  
  # file=streamed_code_LAD_$((scale)).txt
  # ./Streamed_Cuda_Solution $scale ./LAD/images "LAD" $start_frame $end_frame 3 -t vector > MAIN_OUTPUT/STREAM_OUTPUT/$file
  
  # file=streamed_code_NV_$((scale)).txt
  # ./Streamed_Cuda_Solution $scale ./NV/images "NV" $start_frame $end_frame 3 -t vector > MAIN_OUTPUT/STREAM_OUTPUT/$file

done
