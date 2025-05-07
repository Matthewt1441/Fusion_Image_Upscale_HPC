Read me for Image Upscaling Project by Kyler Martinez and Matthew Toro.

----------------------Project Directory-----------------------
After downloading the project and transferring onto the HPC there should the primary folder "Fusion_Image_Upscale_HPC", we preferred to place this folder in the ece569 folder but any location will work as long as the proper paths are set later on.

Inside are two primary folders, the "build_dir" and the "labs" folder. The build_dir is where the input data, job slurm files, executables, and all output images and files. The labs directory contains all the source code for the test benches including all the files needed to compile the project.

-------------------------Input Data---------------------------
There are three datasets to choose from:
- LM: Luigi's Mansion (2005) rendered at 640 x 480p
-- File Format: ./LM/images

- LAD: Like a Dragon Infinite Wealth (2024) rendered at 1280 x 720p
-- File Format: ./LAD/images

- NV: Fallout: New Vegas (2010) rendered at 1920 x 1080p
-- File Format: ./NV/images

The datasets contain a subset of the input set we used for evaluation, the images require a lot of storage due to the PPM file format and take a long time to transfer to the HPC so a small amount of frames were included.

**See the end of the document for information on making your own dataset**

------------------------Compiling Code------------------------
To compile the project, open a command window and use the cd command to enter the build_dir directory, alternatively you can open the command line in the folder using the GUI. To compile the project please execute the following commands:

1. module load cuda11/11.0
- loads the CUDA module to compile code and use profiling tools

2. CC=gcc cmake ../labs
- Creates the make files for the project. Note: we have provided the make files, but this is here just in case.

3. make clean
- removes previously compiled code

4. make
- creates all the executables

-------------------------Test Benches-------------------------
The labs folder contains our three primary test benches:

1. Serial Code Test Bench
2. Optimized Test Bench
3. Stream Test Bench

These test benches show the evolution of our project. 

It is important that the serial test bench executes before any of the other test benches because its output files are needed by the CUDA test benches for verification. We include the serial output images but if the files are changed then this order must be followed or errors will occur.

----------------------Executing a Solution--------------------
The solutions use similiar arguments and the test benches use the following argument structures

./Serial_Fusion_Solution {SCALE} {FILE FORMAT} {VIDEO GAME NAME} {START FRAME} {END FRAME}
./Optimized_Cuda_Solution {SCALE} {FILE FORMAT} {VIDEO GAME NAME} {START FRAME} {END FRAME}
./Streamed_Cuda_Solution {SCALE} {FILE FORMAT} {VIDEO GAME NAME} {START FRAME} {END FRAME} {STREAM COUNT}

SCALE: Integer scale value for upscaling. 			Example: 2
FILE FORMAT: File path and file name format for the input data. Example: ./NV/images
VIDEO GAME NAME: Name for the video game. 			Example: Fallout New Vegas
START FRAME: Frame value to start at. 				Example 1
END FRAME: Frame value to stop at. 				Example: 15
STREAM COUNT: The desired number of streams, 3 worked the best. Example: 3

Using the terminal that is currently in the build_dir, you can use this command structure to start the executable. An example is provided below:
./Streamed_Cuda_Solution 2 ./NV/images NV 1 15 3

After executing one of the solutions, the terminal will display timing information to show the framerate the solution executed at. Additionally new files have been generated in build_dir/MAIN_OUPUT/SERIAL_OUPUT, OPTIMIZED_OUTPUT, or STREAM_OUTPUT depending on the solution selected. 

-----------------------Submitting a Job-----------------------
To submit a job for execution please use the two provided .slurm files.
1. main.slurm
- Contains all CUDA solutions and upscales 15 frames for all datasets at a subsect of scale factors. The stream size is set to 3.
2. Serial_Only.slurm
- Contains the Serial solution and upscales 15 frames for all datasets at a subsect of scale factors

****It is crucial that the Serial_Only slurm is executed before the main.slurm. It takes a long time to execute so we provided the output images, so it is not required to submit the job. The serial code is incredibly slow and there is not much we can do to resolve this but the GPU code is much faster.****

To submit the jobs you first need to access the files and modify line 34 and change the path for the build_dir to match your system. Failure to do this will result in the slurm not finding the executables and input data. Additional scale factors can be commented out by using #.

Once the files are ready, return to the terminal that is using the build_dir and type the following commands:
 - srun Serial_Only.slurm
 - srun main.slurm

There should now be plenty of files in the MAIN_OUTPUT folder and we will discuss the types in the next section.

---------------Project Verification & Analysis----------------
Output files will be saved in:
	build_dir/MAIN_OUPUT/SERIAL_OUPUT
	build_dir/MAIN_OUPUT/OPTIMIZED_OUTPUT
	build_dir/MAIN_OUPUT/STREAM_OUTPUT

In each folder there will be two types of files, the job output and the upscaled image.

Serial and Optimized images will be in the format: FUSED_Frame{END_FRAME}_Scale_{SCALE}_Game_{GAME NAME}.ppm
Streamed solution images will be in the format: FUSED_Frame{END_FRAME}_Streams_{STREAM_COUNT}_Scale_{SCALE}_Game_{GAME_NAME}_StreamId_{STREAM_COUNT-1}.ppm

Output files with timing have the following format: {SOLUTION TYPE}_code_{GAME NAME}_{SCALE}.txt

For example, the streamed solution example from the "Executing a Solution" section would have the following files in the STREAM_OUTPUT folder:
- Image = FUSED_Frame15_Streams_3_Scale_2_Game_NV_StreamId_2.ppm
- Job Output = streamed_code_NV_2.txt

The images can be compared with the corresponding input image and see that image is a larger version and the image dimensions in the bottom corner are scaled up. The serial and CUDA upscaled images can be comapred side by side but it will be difficult to notice the differences with the naked eye.

The job output files contain frame rate timing for each frame and overall timing at the end. To better compare each solution it is recommended to go to the bottom of output file and compare the overall frame rate and execution time.

- FINAL SERIAL CODE FPS, 0.320563, Total Execution Time for 15 frames, 46.825000, s

The CUDA solutions have two frame rates listed, the effective/total FPS and the compute FPS. The compute FPS only times the computation portion of the test bench and excludes malloc times. This is to help better compare solutions because the streamed solution requires more setup time but its computation FPS is greater than that of a single stream.

- FINAL CUDA CODE COMPUTE FPS, 247.029037, Total Execution Time for 15 frames, 0.006462, s
- FINAL CUDA CODE FPS, 90.350441, Total Execution Time for 15 frames, 0.006462, s
- Accuracy: 99.103409%

- FINAL CUDA STREAM CODE, STREAMS, 3 COMPUTE FPS, 267.857143, Compute Execution Time for 15 frames, 0.056000, s
- FINAL CUDA STREAM CODE, STREAMS, 3 EFFECTIVE FPS, 92.592593, Total Execution Time for 15 frames, 0.162000, s
- Accuracy: 99.103409%

Aside from the visual inspection and timing, we can functionally verify the CUDA versions of the algorithm by an accuracy rating. We compute this by summing the difference between the serial image and CUDA output image and averageing it by the maximum error that could be achieved across the image. Due to this, the serial image must be avaliable before running any CUDA test benches. That is the extent of the analysis found in our source code, our presentation and paper have more information and further analysis.


------------Thank You For Exploring Our Demo and Enjoy!-------

------------Extra Notes: Creating Your Own Dataset------------
To create your own PPM dataset:

1. Download the FFMPEG executable.
2. Place in a directory where you want to store the video game recording.
3. Aquire a recording of your game, some emulators like Dolphin allow you to export frames directly as an .AVI file or you can screen record but may be limited on the framerate you can capture.
4. Type the following cammand and change the paramters:
	./ffmpeg -i input_video.avi -ss START TIME (MM:SS) -to END TIME (MM:SS) output_images%1d.ppm
 - input_video.avi is the name of the recording
 - START TIME is the first frame to capture, write this in the format MM:SS like 00:30 to start 30 seconds in.
 - END TIME is the last frame to capture and is in the same format as the start time
 - output_images%1d.ppm will create files with the name output_images followed by the frame number represented by %1d.
