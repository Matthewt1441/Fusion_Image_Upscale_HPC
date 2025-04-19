// do not modify this file for histogram versions 0 and 1
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>

#include <chrono>

#include "kernel.cu"
#include "serial_code.cu"

int main(int argc, char* argv[])
{

    int scale = atoi(argv[1]); int block_dim = atoi(argv[2]);

    int big_width, width;
    int big_height, height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char*  h_BIC;
    unsigned char* h_NN;
    unsigned char* h_img;
    unsigned char*  h_img_nn_grey;                    //Reference NN Picture
    unsigned char*  h_img_bic_grey;                   //Reference BIC Picture
    float*          h_diff_map;                           //Upscaled ARTIFACT
    float*          h_ssim_map;                           //Upscaled ARTIFACT
    float*          h_artifact_map;                           //Upscaled ARTIFACT


    unsigned char* h_ARTI_gray;

    //Device Array Pointers
    unsigned char* d_img_nn_grey;                    //Reference NN Picture
    unsigned char* d_img_bic_grey;                   //Reference BIC Picture
    float* d_naive_diff_map;
    float* d_naive_ssim_map;
    float* d_naive_artifact_map;
    float* d_basic_artifact_map;
    float* d_atcs_artifact_map;
    float* d_otcs_artifact_map;
    float* d_mw_artifact_map;
    char file_name[50];

    //Lets start off with timing one image
    try
    {
        //Check that CUDA-capable GPU is installed
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }


        double processing_time = 0;

         //Load Input Image

         //Read in first image initially to get input width and height.
        //sprintf(file_name, "./NV/image%d.ppm", 1);
        //sprintf(file_name, "./LAD/LAD_%d.ppm", 1);
        sprintf(file_name, "./LM_Frame/image%d.ppm", 1);
        h_img = (unsigned char*)readPPM(file_name, &width, &height);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        h_NN =  (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
        h_BIC = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
        h_img_nn_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
        h_img_bic_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
        h_ARTI_gray = (unsigned char*) malloc(sizeof(unsigned char) * big_width * big_height);

        h_diff_map = (float*) malloc(sizeof(float) * big_width * big_height);
        h_ssim_map = (float*) malloc(sizeof(float) * big_width * big_height);
        h_artifact_map = (float*) malloc(sizeof(float) * big_width * big_height);

        //******** Malloc Device Images ********//

        //Original BIC & NN
        if (cudaMalloc((void**)&d_img_nn_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_img_bic_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Gray Pictures
        if (cudaMalloc((void**)&d_naive_diff_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "NAIVE DIFF MAP Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_naive_ssim_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "SSIM MAP Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_naive_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "ARTI NAIVE Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_basic_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "ARTI BASIC Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_atcs_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "ARTI NAIVE Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_otcs_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "ARTI BASIC Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_mw_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "ARTI BASIC Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //**************** Setup Kernel ****************//
        dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block(block_dim, block_dim);

        dim3 Grid_Arti(((big_width - 1) / 8) + 1, ((big_height - 1) / 8) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block_Arti(8, 8);

        dim3 Grid_Arti2(((big_width - 1) / 16) + 1, ((big_height - 1) / 16) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block_Arti2(16, 16);

        //**************** Setup Kernel ****************//

        //Variables for timing
        cudaEvent_t astartEvent, astopEvent;
        float aelapsedTime;
        cudaEventCreate(&astartEvent);
        cudaEventCreate(&astopEvent);

        //Copy Input Image to Device
        nearestNeighbors(h_img_nn_grey, h_NN, big_width, big_height, h_img, width, height, scale);
        bicubicInterpolation(h_BIC, big_width, big_height, h_img, width, height, scale);
        RGB2Greyscale(h_img_bic_grey, h_BIC, big_width, big_height);

        cudaMemcpy(d_img_nn_grey, h_img_nn_grey, sizeof(unsigned char) * big_width * big_height, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img_bic_grey, h_img_bic_grey, sizeof(unsigned char) * big_width * big_height, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        //******************************* Run & Time Kernels ********************************//
        printf("ARTIFACT Test\nBlock Dimensions, %d, Image Dimensions, %d, %d\n", block_dim, big_width, big_height);

        ////////////TIME ARTIFACT WITH SERIAL CODE IMPLEMENTATION/////////////////////
        auto start = std::chrono::high_resolution_clock::now();

        ABS_Difference_Grey(h_diff_map, h_img_nn_grey, h_img_bic_grey, big_width, big_height);
        SSIM_Grey(h_ssim_map, h_img_nn_grey, h_img_bic_grey, big_width, big_height);
        MapMul(h_artifact_map, h_diff_map, h_ssim_map, big_width, big_height);

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - start;

        processing_time = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

        printf("ARTI SERIAL CODE, %f, ms\n", processing_time/1000);

        //WRITE THE PICTURES FOR COMPARISON
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;
        writePPMGrey("./ARTI_TEST/ARTI_SERIAL.ppm", (char*)h_ARTI_gray, big_width, big_height);

        cudaDeviceSynchronize();

        ////////////TIME ARTIFACT WITH NAIVE CUDA IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        ABS_Difference_Grey_Kernel << < Grid, Block >> > (d_naive_diff_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);
        SSIM_Grey_Kernel << < Grid, Block >> > (d_naive_ssim_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);

        cudaDeviceSynchronize();
        MapMulKernel << < Grid, Block >> > (d_naive_artifact_map, d_naive_diff_map, d_naive_ssim_map, big_width, big_height);
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("ARTI NAIVE CUDA, %f, ms\n", aelapsedTime);

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_artifact_map, d_naive_artifact_map, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;

        //WRITE THE PICTURES FOR COMPARISON
        writePPMGrey("./ARTI_TEST/ARTI_NAIVE.ppm", (char*)h_ARTI_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME ARTIFACT WITH BASIC OPTIMIZED IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        Artifact_Grey_Kernel << < Grid, Block >> > (d_basic_artifact_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("ARTI BASIC OPTIMIZED, %f, ms\n", aelapsedTime);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_artifact_map, d_basic_artifact_map, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;

        //WRITE THE PICTURES FOR COMPARISON
        writePPMGrey("./ARTI_TEST/ARTI_BASIC.ppm", (char*)h_ARTI_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME ARTIFACT WITH SHARED MEM ALL THREADS CALCULATE SSIM (ATCS) IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);
        Artifact_Shared_Memory_Kernel << < Grid_Arti, Block_Arti, sizeof(float) * 2 * 8 * 8 >> > (d_atcs_artifact_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("ARTI ATCS, %f, ms\n", aelapsedTime);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_artifact_map, d_atcs_artifact_map, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;

        //WRITE THE PICTURES FOR COMPARISON
        writePPMGrey("./ARTI_TEST/ARTI_ATCS.ppm", (char*)h_ARTI_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME ARTIFACT WITH SHARED MEM ONE THREAD CALCULATE SSIM (OTCS) IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        Artifact_Shared_Memory_Kernel2 << < Grid_Arti, Block_Arti, sizeof(float) *  2 * 8 * 8 >> > (d_otcs_artifact_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("ARTI OTCS, %f, ms\n", aelapsedTime);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_artifact_map, d_otcs_artifact_map, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;

        //WRITE THE PICTURES FOR COMPARISON
        writePPMGrey("./ARTI_TEST/ARTI_OTCS.ppm", (char*)h_ARTI_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME ARTIFACT WITH SHARED MEM MULTIPLE WINDOWS (MW) IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        Artifact_Shared_Memory_Kernel3 << < Grid_Arti2, Block_Arti2, sizeof(float) * 2*16 * 16 >> > (d_mw_artifact_map, d_img_nn_grey, d_img_bic_grey, big_width, big_height);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("ARTI MMW, %f, ms\n", aelapsedTime);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        //COPY OVER IMAGE DATA
        cudaMemcpy(h_artifact_map, d_mw_artifact_map, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);
        Map2Greyscale(h_ARTI_gray, h_artifact_map, big_width , big_height, 255); //Artifact values should be between 0-255;

        //WRITE THE PICTURES FOR COMPARISON
        writePPMGrey("./ARTI_TEST/ARTI_MW.ppm", (char*)h_ARTI_gray, big_width, big_height);
        cudaDeviceSynchronize();

        //************************* CHECK FOR CORRECTNESS *****************************//
        printf("\n\nCHECKING FOR CORRECTNESS\n");

        //Compare with Serial Image
        unsigned char* serial_check_img_gray        = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_SERIAL.ppm", &big_width, &big_height);

        //TEST 1: NAIVE CUDA
        unsigned char* naive_check_img_gray         = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_NAIVE.ppm", &big_width, &big_height);
        printf("CHECKING NAIVE CUDA KERNEL");
        Grey_Image_Compare(serial_check_img_gray, naive_check_img_gray, big_width, big_height);

        //TEST 2: BASIC OPTIMIZED CUDA
        unsigned char* basic_check_img_gray         = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_BASIC.ppm", &big_width, &big_height);
        printf("CHECKING BASIC OPTIMIZED CUDA KERNEL");
        Grey_Image_Compare(serial_check_img_gray, basic_check_img_gray, big_width, big_height);

        //TEST 3: SHARED MEMORY ATCS CUDA
        unsigned char* shared_atcs_check_img_gray   = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_ATCS.ppm", &big_width, &big_height);
        printf("CHECKING ATCS SOTCSARED MEM");
        Grey_Image_Compare(serial_check_img_gray, shared_atcs_check_img_gray, big_width, big_height);

        //TEST 4: SHARED MEMORY OTCS MEM
        unsigned char* shared_otcs_check_img_gray   = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_OTCS.ppm", &big_width, &big_height);
        printf("CHECKING MW SHARED MEM");
        Grey_Image_Compare(serial_check_img_gray, shared_otcs_check_img_gray, big_width, big_height);

        //TEST 5: SHARED MEMORY MW CUDA
        unsigned char* shared_mw_check_img_gray   = (unsigned char*)readPPMGray("./ARTI_TEST/ARTI_MW.ppm", &big_width, &big_height);
        printf("CHECKING MW SHARED MEM");
        Grey_Image_Compare(serial_check_img_gray, shared_mw_check_img_gray, big_width, big_height);
    
        free(serial_check_img_gray);        free(naive_check_img_gray);         free(basic_check_img_gray);
        free(shared_atcs_check_img_gray);   free(shared_otcs_check_img_gray);   free(shared_mw_check_img_gray);

        //************************* CLEAN UP *****************************//
        // 
        // TODO ACTUALLY FREE THE MEMORY LOL
        //Free Host Memory
        free(h_img);
        free(h_img_nn_grey);                    free(h_img_bic_grey);  

        //Free device Memory
        cudaFree(d_img_nn_grey);                cudaFree(d_img_bic_grey);

        cudaFree(d_naive_diff_map);             cudaFree(d_naive_ssim_map);
        cudaFree(d_naive_artifact_map);         cudaFree(d_basic_artifact_map);

        cudaFree(d_atcs_artifact_map);          cudaFree(d_otcs_artifact_map);
        cudaFree(d_mw_artifact_map);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    catch (const char* err)
    {
        printf("%s\n", err);
        return 1;
    }

    cudaDeviceReset();
    return 0;
}