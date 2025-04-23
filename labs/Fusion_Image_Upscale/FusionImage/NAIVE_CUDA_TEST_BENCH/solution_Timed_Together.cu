// Final Algorithm Set For Fusion Upscaling
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>

#include "util.cu"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    // ARGUMENTS (Example Below)
    // ./CUDA_CODE_TEST_BENCH_Solution 2 ./NV/image "Fallout New Vegas" 1 10
    // EXECUTABLE, SCALE, FILE_FORMAT, GAME_NAME, START_FRAME, END_FRAME
    //          0,     1,           2,         3,           4,         5

    char file_path[50];
    char file_name[50];
    char file_address[50];

    char game_name[50];

    int scale = atoi(argv[1]); 
            
    strcpy(file_path, argv[2]);     //copy file path into the var 
    strcpy(game_name, argv[3]);     //copy game name into the var

    int start_frame = atoi(argv[4]);
    int end_frame   = atoi(argv[5]);
    int current_frame = start_frame;
    
    memset(file_address, 0, sizeof(file_address));

    strcat(file_address, file_path); 
    sprintf(file_name, "%d.ppm", start_frame);
    strcat(file_address, file_name);

    printf(file_address); printf("\n");

    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char*  h_img;                              //Original Small Input Image
    unsigned char*  h_big_img_fused;                    //Upscaled Fused Image

    //Device Array Pointers
    unsigned char* img_cuda;
    unsigned char* big_img_nn_cuda;
    unsigned char* big_img_bic_cuda;
    unsigned char* big_img_nn_grey_cuda;
    unsigned char* big_img_bic_grey_cuda;

    float* big_diff_map_cuda;
    float* big_ssim_map_cuda;
    float* big_artifact_map_cuda;
    float* big_artifact_blurred_map_cuda;
    

    //Lets start off with timing one image
    try
    {
        //Check that CUDA-capable GPU is installed
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }


        float processing_time = 0;
 
        //**************** Setup Kernel ****************//
        h_img = (unsigned char*)readPPM(file_address, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        //******** Malloc Host Images ********//
        h_big_img_fused                 = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        //******** Malloc Host Images ********//


        //******** Malloc Device Images ********//
        //Original Image
        if (cudaMalloc((void**)&img_cuda, const_width * const_height * sizeof(unsigned char) * 3) != cudaSuccess)
            printf("Original Image Failed To Copy To Device.\n");      //Notify failure

        //Upscaled Images
        if (cudaMalloc((void**)&big_img_nn_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_img_bic_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Grey Versions for Upscaled Images
        if (cudaMalloc((void**)&big_img_nn_grey_cuda, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_img_bic_grey_cuda, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Maps for Fusion
        if (cudaMalloc((void**)&big_diff_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Grey Difference Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_ssim_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "SSIM Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_artifact_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&big_artifact_blurred_map_cuda, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Final Image
        if (cudaMalloc((void**)&big_img_fused_cuda, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        //******** Malloc Device Images ********//
        cudaError_t error;

        //**************** Setup Kernel ****************//

        //Variables for timing
        cudaEvent_t total_Time_Start, total_Time_End, one_Frame_Start, one_Frame_End, compute_Frame_Start, compute_Frame_End;

        cudaEventCreate(&total_Time_Start);         cudaEventCreate(&total_Time_End);        
        cudaEventCreate(&one_Frame_Start);          cudaEventCreate(&one_Frame_End);
        cudaEventCreate(&compute_Frame_Start);      cudaEventCreate(&compute_Frame_End);

        float execution_time_ms = 0, execution_time_s;
        float current_fps = 0;
        float average_execution_time = 0, average_execution_time_compute = 0;
        float average_fps = 0;

        //******************************* Run & Time CODE ********************************//
        printf("NAIVE CUDA Code Test\n");
        printf("Scale Factor, %d, Game, %s, Start Frame, %d, End Frame, %d\n", scale, game_name, start_frame, end_frame);
        printf("Input Image Dimensions, %d , %d, Output Image Dimensions, %d, %d\n", width, height, big_width, big_height);

        cudaEventRecord(total_Time_Start, 0);           //BEGIN TOTAL EXECUTION TIME

        dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block(block_dim, block_dim);

        int const_width = width;
        int const_height = height;

        while (current_frame <= end_frame)
        {
            cudaEventRecord(one_Frame_Start, 0);        //BEGIN COMPLETE FRAME TIMING

            //PHASE 0 : Load Input Image
            memset(file_address, 0, sizeof(file_address));

            strcat(file_address, file_path); 
            sprintf(file_name, "%d.ppm", current_frame);
            strcat(file_address, file_name);
            //printf(file_address); printf("\n");

            h_img = (unsigned char*)readPPM(file_address, &width, &height);

            //Copy Input Image to Device
            cudaMemcpy(img_cuda, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            //Launch the kernel and pass device matricies and size information
            nearestNeighborsKernel <<< Grid, Block >> > (big_img_nn_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);
            bicubicInterpolationKernel <<< Grid, Block >> > (big_img_bic_cuda, img_cuda, big_width, big_height, const_width, const_height, scale);

            //cudaDeviceSynchronize();
            
            //NOTE: TECHNICALLY RUNS FASTER IF RGB A SEPERATE KERNEL FOR NAIVE IMPLEMENTATION
            RGB2GreyscaleKernel <<< Grid, Block >>> (big_img_nn_cuda, big_img_nn_grey_cuda, big_width, big_height);
            RGB2GreyscaleKernel <<< Grid, Block >>> (big_img_bic_cuda, big_img_bic_grey_cuda, big_width, big_height);

            //cudaDeviceSynchronize();

            ABS_Difference_Grey_Kernel <<< Grid, Block >>> (big_diff_map_cuda, big_img_nn_grey_cuda, big_img_bic_grey_cuda, big_width, big_height);
            SSIM_Grey_Kernel <<< Grid, Block >>> (big_ssim_map_cuda, big_img_nn_grey_cuda, big_img_bic_grey_cuda, big_width, big_height);

            //cudaDeviceSynchronize();

            MapMulKernel <<< Grid, Block >>> (big_artifact_map_cuda, big_diff_map_cuda, big_ssim_map_cuda, big_width, big_height);

            //cudaDeviceSynchronize();

            GuassianBlur_Map_Kernel <<< Grid, Block >>> (big_artifact_blurred_map_cuda, big_artifact_map_cuda, big_width, big_height, 3, 1.5);

            //cudaDeviceSynchronize();

            MapThreshold_Kernel <<< Grid, Block >>> (big_artifact_blurred_map_cuda, 0.05, big_width, big_height);

            //cudaDeviceSynchronize();

            Image_Fusion_Kernel << < Grid, Block >> > (big_img_fused_cuda, big_img_nn_cuda, big_img_bic_cuda, big_artifact_blurred_map_cuda, big_width, big_height);

            cudaDeviceSynchronize();

            cudaMemcpy(big_img_fused, big_img_fused_cuda, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);

            cudaEventRecord(compute_Frame_End, 0);
            cudaEventSynchronize(compute_Frame_End);
            
            // error = cudaGetLastError();
            // if(error != cudaSuccess)
            // {
            //     // print the CUDA error message and exit
            //     printf("RGBA -> RGB CUDA error: %s\n", cudaGetErrorString(error));
            //     exit(-1);
            // }

            //Send Device Images to Host
            cudaDeviceSynchronize();
            cudaMemcpy(h_big_img_fused, big_img_fused, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
            cudaEventRecord(one_Frame_End, 0);
            cudaEventSynchronize(one_Frame_End);
            
            cudaEventElapsedTime(&processing_time, compute_Frame_Start, compute_Frame_End);

            execution_time_ms = processing_time;
            execution_time_s = execution_time_ms / 1000;
            current_fps = 1/(execution_time_s);
            average_execution_time_compute += execution_time_s;
            average_fps = 1/(average_execution_time_compute / (current_frame - start_frame + 1));

            printf("CUDA CODE COMPUTE: FRAME, %d, TIME, %f, ms, CURRENT FPS, %f, AVERAGE FPS, %f\n", current_frame, execution_time_ms, current_fps, average_fps);

            cudaEventElapsedTime(&processing_time, one_Frame_Start, one_Frame_End);
            execution_time_ms = processing_time;
            execution_time_s = execution_time_ms / 1000;
            current_fps = 1/(execution_time_s);
            average_execution_time += execution_time_s;
            average_fps = 1/(average_execution_time / (current_frame - start_frame + 1));

            printf("CUDA CODE TOTAL: FRAME, %d, TIME, %f, ms, CURRENT FPS, %f, AVERAGE FPS, %f\n", current_frame, execution_time_ms, current_fps, average_fps);

            current_frame++;
        }
        cudaEventRecord(total_Time_End, 0);
        cudaEventSynchronize(total_Time_End);
        cudaEventElapsedTime(&processing_time, compute_Frame_Start, total_Time_End);
        execution_time_s = processing_time / 1000;

        int     total_frames    = end_frame - start_frame + 1;
        double  compute_fps     = total_frames / average_execution_time_compute;
        double  total_fps       = total_frames / average_execution_time;

        printf("\n");
        printf("FINAL CUDA CODE COMPUTE FPS, %f, Total Execution Time for %d frames, %f, s\n", compute_fps, end_frame - start_frame + 1, execution_time_s);
        printf("FINAL CUDA CODE FPS, %f, Total Execution Time for %d frames, %f, s\n", average_fps, end_frame - start_frame + 1, execution_time_s);

        memset(file_address, 0, sizeof(file_address));

        strcat(file_address, "./CUDA_OUTPUT_TOGETHER/FUSED_"); 

        sprintf(file_name, "Scale_%d_Game_%s", scale, game_name);

        strcat(file_address, file_name);
        strcat(file_address, ".ppm");

        writePPM(file_address, (char*)h_big_img_fused, big_width, big_height);

        //************************* CLEAN UP *****************************//
        // 
        //Free Host Memory
        free(h_img);
        free(h_big_img_fused);

        //Free device Memory
        cudaFree(d_img);
        cudaFree(d_RGBA_img);
        cudaFree(d_big_img_nn);
        cudaFree(d_big_img_bic);
        cudaFree(d_big_img_nn_grey);
        cudaFree(d_big_img_bic_grey);
        cudaFree(d_big_artifact_map);
        cudaFree(d_big_blurred_artifact_map);
        cudaFree(d_big_blurred_artifact_map_inter);
        cudaFree(d_big_rgba_img_fused);
        cudaFree(d_big_img_fused);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}