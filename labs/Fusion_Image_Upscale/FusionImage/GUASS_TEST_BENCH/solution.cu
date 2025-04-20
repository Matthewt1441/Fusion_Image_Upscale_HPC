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

    int scale = atoi(argv[1]); int block_dim_y = atoi(argv[2]); int block_dim_x = atoi(argv[3]);
    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char*  h_img;                              //Original Small Input Image
    unsigned char*  h_big_img_nn;                       //Upscaled Nearest Neighbor Image
    unsigned char*  h_big_img_nn_grey;                  //Upscaled Greyscale Nearest Neighbor Image
    unsigned char*  h_big_img_bic;                      //Upscaled Bicubic Image
    unsigned char*  h_big_img_bic_grey;                 //Upscaled Greyscale Bicubic Image
    unsigned char*  h_big_img_DIFF_grey;                //Upscaled Greyscale Difference Image
    unsigned char*  h_big_img_SSIM_grey;                //Upscaled Greyscale SSIM Image
    unsigned char*  h_big_img_ARTIFACT_grey;            //Upscaled Greyscale ARTIFACT Image
    unsigned char*  h_big_img_BLURRED_ARTIFACT_grey;    //Upscaled Greyscale BLURRED ARTIFACT Image
    unsigned char*  h_big_img_fused;                    //Upscaled Fused Image
    float*          h_diff_map;                         //Difference Map
    float*          h_ssim_map;                         //SSIM Map
    float*          h_artifact_map;                     //Artifact Map
    float*          h_blurred_artifact_map;             //Blurred Artifact Map

    //Device Array Pointers
    unsigned char*      d_img;                              //Original Small Input Image
    RGBA_t*             d_RGBA_img;                         //Original Small Input Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_nn;                       //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_bic;                      //Upscaled Bicubic Image w/ 32bit-pixel format
    unsigned char*      d_big_img_nn_grey;                  //Upscaled Greyscale Nearest Neighbor Image
    unsigned char*      d_big_img_bic_grey;                 //Upscaled Greyscale Bicubic Image
    float*              d_big_artifact_map;                 //Upscaled Artifact Map for image fusion
    float*              d_big_blurred_artifact_map;         //Upscaled Blurred Artifact Map for image fusion
    float*              d_big_blurred_artifact_map_inter;   //Upscaled Blurred Artifact Map for image fusion
    RGBA_t*             d_big_rgba_img_fused;               //Upscaled Fused Image w/ 32bit-pixel format
    unsigned char*      d_big_img_fused;                    //Upscaled Fused Image  
    

    //Lets start off with timing one image
    try
    {
        //Check that CUDA-capable GPU is installed
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        //***** Temp *****//
        char file_path[50];
        char file_name[50];
        int count = 0;

        int current_img = 1;

        double processing_time = 0;
 
        //**************** Setup Kernel ****************//
        
        strcpy(file_name, argv[4]); // copy string one into the result.
        strcat(file_name, "1.ppm"); // append string two to the result.
        
        printf(file_name);
        printf("\n");

        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        //******** Malloc Host Images ********//
        h_big_img_nn                    = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_nn_grey               = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_bic                   = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_big_img_bic_grey              = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_ARTIFACT_grey         = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_BLURRED_ARTIFACT_grey = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count);
        h_big_img_fused                 = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_diff_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_ssim_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_artifact_map                  = (float*)malloc(sizeof(float) * big_pixel_count);
        h_blurred_artifact_map          = (float*)malloc(sizeof(float) * big_pixel_count);
        //******** Malloc Host Images ********//


        //******** Malloc Device Images ********//
        //Original Image & RGBA Image
        if (cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_RGBA_img, width * height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Grey Versions for Upscaled Images
        if (cudaMalloc((void**)&d_big_img_nn_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_img_bic_grey, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Maps for Fusion
        if (cudaMalloc((void**)&d_big_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_blurred_artifact_map_inter, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Intermediate Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_blurred_artifact_map, big_width * big_height * sizeof(float)) != cudaSuccess)
            fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Final Image and RGBA Image
        if (cudaMalloc((void**)&d_big_img_fused, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_big_rgba_img_fused, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        //******** Malloc Device Images ********//

        dim3 RGB_Block(256);
        dim3 RGB_Grid(ceil((big_width * big_height) / (float)RGB_Block.x));
        int  rgbToRGBA_Shared_Mem_Size = sizeof(unsigned char)*RGB_Block.x * 3;
        int  rgbaToRGB_Shared_Mem_Size = sizeof(RGBA_t)*RGB_Block.x;

        dim3 NN_Block(16, 16);
        dim3 NN_Grid(((big_width - 1) / NN_Block.x) + 1, ((big_height - 1) / NN_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
 
        dim3 BiCubic_Block(4*scale, 4*scale);
        dim3 BiCubic_Grid(((big_width - 1) / BiCubic_Block.x) + 1, ((big_height - 1) / BiCubic_Block.y) + 1);
        int  BiCubic_Shared_Mem_Size = sizeof(RGBA_t) * ((BiCubic_Block.y / scale) + 3) * ((BiCubic_Block.x / scale) + 3);
        
        dim3 Arti_Block(8, 8);
        dim3 Arti_Grid(((big_width - 1) / Arti_Block.x) + 1, ((big_height - 1) / Arti_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        int  Arti_Shared_Mem_Size = sizeof(float) * 2 * 8 * 8;

        //Setup Guassian Blur based on passed in Args
        int GUAS_Ksize = 7;
        float GUAS_Sigma = 1.5;
        dim3 h_Guas_Block(block_dim_x, block_dim_y);
        dim3 h_Guas_Grid(((big_width - 1) / h_Guas_Block.x) + 1, ((big_height - 1) / h_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 v_Guas_Block(block_dim_x, block_dim_y);
        dim3 v_Guas_Grid(((big_width - 1) / v_Guas_Block.x) + 1, ((big_height - 1) / v_Guas_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Gaus_Naive_Block(block_dim_x, block_dim_y);
        dim3 Gaus_Naive_Grid(((big_width - 1) / Gaus_Naive_Block.x) + 1, ((big_height - 1) / Gaus_Naive_Block.y) + 1); 



        cudaError_t error;
        //**************** Setup Kernel ****************//

        //Variables for timing
        cudaEvent_t astartEvent1, astopEvent1, astartEvent2, astopEvent2;
        float aelapsedTime1, aelapsedTime2;
        cudaEventCreate(&astartEvent1);
        cudaEventCreate(&astopEvent1);        
        cudaEventCreate(&astartEvent2);
        cudaEventCreate(&astopEvent2);

        //Load Input Image
        h_img = (unsigned char*)readPPM(file_name, &width, &height);

        //Copy Input Image to Device
        cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        //Convert original image to RGBA image
        rgbToRGBA_Kernel << < RGB_Grid, RGB_Block, rgbToRGBA_Shared_Mem_Size >> > (d_RGBA_img, d_img, width * height);
        cudaDeviceSynchronize();

        //Serial Code Up Until Guass
        nearestNeighbors(h_big_img_nn_grey, h_big_img_nn, big_width, big_height, h_img, width, height, scale);
        bicubicInterpolation(h_big_img_bic, big_width, big_height, h_img, width, height, scale);
        RGB2Greyscale(h_big_img_bic_grey, h_big_img_bic, big_width, big_height);
        
        ABS_Difference_Grey(h_diff_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
        SSIM_Grey(h_ssim_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
        MapMul(h_artifact_map, h_diff_map, h_ssim_map, big_width, big_height);

        //CUDA Code Up until Guass
        nearestNeighbors_GreyCon_Kernel_RGBA                    <<< NN_Grid, NN_Block >>>                                   (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA  <<< BiCubic_Grid, BiCubic_Block, BiCubic_Shared_Mem_Size >>>(d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);
        Artifact_Shared_Memory_Kernel                           <<< Arti_Grid, Arti_Block, Arti_Shared_Mem_Size >>>         (d_big_artifact_map, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);
        //Artifact_Grey_Kernel                                    <<< Arti_Grid, Arti_Block >>>                               (d_big_artifact_map         , d_big_img_nn_grey             , d_big_img_bic_grey        , big_width, big_height);
        
        //******************************* Run & Time Kernels ********************************//
        printf("Guassian Blur Test\nBlock Dimensions, %d x %d, Scale Factor, %d\nInput Image Dimensions, %d , %d\nOutput Image Dimensions, %d, %d\n", block_dim_x, block_dim_y, scale, width, height, big_width, big_height);
        #define ITERATIONS 5
        double Serial_Time[ITERATIONS] = {0};
        double Naive_Time[ITERATIONS] = {0};
        double Horizontal_Seperable_Time[ITERATIONS] = {0};
        double Vertical_Seperable_Time[ITERATIONS] = {0};
        double Average_Time = 0;
        for (int i = 0; i < ITERATIONS; i ++)
        {
            ////////////TIME GUASSIAN WITH SERIAL CODE IMPLEMENTATION/////////////////////
            auto start = std::chrono::high_resolution_clock::now();

            GuassianBlur_Map(h_blurred_artifact_map, h_artifact_map, big_width, big_height, 3, 1.5);
            MapThreshold(h_blurred_artifact_map, 0.05, big_width, big_height);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;
            Serial_Time[i] = std::chrono::duration_cast<std::chrono::microseconds>(dur).count()/1000.0;

            //printf("GUAS SERIAL CODE, %f, ms\n", processing_time/1000);

            //WRITE THE PICTURES FOR COMPARISON
            // if(i == ITERATIONS-1)
            // {
            //     Map2Greyscale(h_big_img_BLURRED_ARTIFACT_grey   , h_blurred_artifact_map, big_width, big_height, 255);   //Artifact values should be between 0-255;
            //     writePPMGrey("./GUAS_TEST/GUAS_SERIAL.ppm", (char*)h_big_img_BLURRED_ARTIFACT_grey, big_width, big_height);
            //     cudaDeviceSynchronize();
            // }
            ////////////TIME GUASSIAN WITH SERIAL CODE IMPLEMENTATION/////////////////////
            
            ////////////TIME GUASSIAN NAIVE CUDA IMPLEMENTATION/////////////////////
            cudaEventRecord(astartEvent1, 0);
            GuassianBlur_Threshold_Map_Naive_Kernel <<< Gaus_Naive_Grid, Gaus_Naive_Block >>>   (d_big_blurred_artifact_map , d_big_artifact_map, big_width, big_height, 3, 1.5, 0.05);
            cudaEventRecord(astopEvent1, 0);
            
            error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("NAIVE CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            cudaDeviceSynchronize();

            cudaEventSynchronize(astopEvent1);
            cudaEventElapsedTime(&aelapsedTime1, astartEvent1, astopEvent1);
            Naive_Time[i] = aelapsedTime1;
            //printf("GUASSIAN NAIVE, %f, ms\n", aelapsedTime1);

            //WRITE THE PICTURES FOR COMPARISON
            // if(i == ITERATIONS-1)
            // {            
            //     //COPY OVER IMAGE DATA
            //     cudaMemcpy(h_blurred_artifact_map   , d_big_blurred_artifact_map, sizeof(float) * big_width * big_height    , cudaMemcpyDeviceToHost);
            //     Map2Greyscale(h_big_img_BLURRED_ARTIFACT_grey   , h_blurred_artifact_map, big_width, big_height, 255);   //Artifact values should be between 0-255;
            //     writePPMGrey("./GUAS_TEST/GUAS_NAIVE.ppm", (char*)h_big_img_BLURRED_ARTIFACT_grey, big_width, big_height);
            //     cudaDeviceSynchronize();
            // }
            ////////////TIME GUASSIAN NAIVE CUDA IMPLEMENTATION/////////////////////


            ////////////TIME GUASSIAN WITH SEPERABLE CUDA IMPLEMENTATION/////////////////////
            cudaEventRecord(astartEvent1, 0);
            horizontalGuassianBlurConvolve  <<< h_Guas_Grid, h_Guas_Block, sizeof(float) * (h_Guas_Block.x + GUAS_Ksize - 1) * h_Guas_Block.y >>>(d_big_blurred_artifact_map_inter, d_big_artifact_map, big_width, big_height, GUAS_Ksize);
            cudaEventRecord(astopEvent1, 0);
    
            cudaEventRecord(astartEvent2, 0);
            verticalGuassianBlurConvolve    <<< v_Guas_Grid, v_Guas_Block, sizeof(float) * (v_Guas_Block.y + GUAS_Ksize - 1) * v_Guas_Block.x >>>(d_big_blurred_artifact_map, d_big_blurred_artifact_map_inter, big_width, big_height, 0.05, GUAS_Ksize);
            cudaEventRecord(astopEvent2, 0);
            
            error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("SEPERABLE CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            cudaDeviceSynchronize();

            cudaEventSynchronize(astopEvent1);
            cudaEventElapsedTime(&aelapsedTime1, astartEvent1, astopEvent1);
            Horizontal_Seperable_Time[i] = aelapsedTime1;
            //printf("GUASSIAN HORIZONTAL SEPERABLE, %f, ms\n", aelapsedTime1);
    
            cudaEventSynchronize(astopEvent2);
            cudaEventElapsedTime(&aelapsedTime2, astartEvent2, astopEvent2);
            Vertical_Seperable_Time[i] = aelapsedTime2;
            //printf("GUASSIAN VERTICAL SEPERABLE, %f, ms\n", aelapsedTime2);

            //WRITE THE PICTURES FOR COMPARISON
            // if(i == ITERATIONS-1)
            // {            
            //     //COPY OVER IMAGE DATA
            //     cudaMemcpy(h_blurred_artifact_map   , d_big_blurred_artifact_map, sizeof(float) * big_width * big_height    , cudaMemcpyDeviceToHost);
            //     Map2Greyscale(h_big_img_BLURRED_ARTIFACT_grey   , h_blurred_artifact_map, big_width, big_height, 255);   //Artifact values should be between 0-255;
            //     writePPMGrey("./GUAS_TEST/GUAS_SEPERABLE_CONVOLVE.ppm", (char*)h_big_img_BLURRED_ARTIFACT_grey, big_width, big_height);
            //     cudaDeviceSynchronize();
            // }
            ////////////TIME GUASSIAN WITH SEPERABLE CUDA IMPLEMENTATION/////////////////////

        }

        printf("Version, ");
        for(int i = 0; i<ITERATIONS; i++)
        {
            printf("Iteration %d, ", i);
        }
        printf("Average\n");

        Average_Time = 0;
        printf("Serial, ");
        for(int i = 0; i<ITERATIONS; i++)
        {
            Average_Time += Serial_Time[i];
            printf("%f, ", Serial_Time[i]);
        }
        printf("%f\n", Average_Time/ITERATIONS);

        Average_Time = 0;
        printf("Naive, ");
        for(int i = 0; i<ITERATIONS; i++)
        {
            Average_Time += Naive_Time[i];
            printf("%f, ", Naive_Time[i]);
        }
        printf("%f\n", Average_Time/ITERATIONS);

        Average_Time = 0;
        printf("Horizontal Seperable, ");
        for(int i = 0; i<ITERATIONS; i++)
        {
            Average_Time += Horizontal_Seperable_Time[i];
            printf("%f, ", Horizontal_Seperable_Time[i]);
        }
        printf("%f\n", Average_Time/ITERATIONS);

        Average_Time = 0;
        printf("Vertical Seperable, ");
        for(int i = 0; i<ITERATIONS; i++)
        {
            Average_Time += Vertical_Seperable_Time[i];
            printf("%f, ", Vertical_Seperable_Time[i]);
        }
        printf("%f\n", Average_Time/ITERATIONS);

        //************************* CLEAN UP *****************************//
        // 
        // TODO ACTUALLY FREE THE MEMORY LOL
        //Free Host Memory
        free(h_img);
        free(h_big_img_nn);
        free(h_big_img_bic);   
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