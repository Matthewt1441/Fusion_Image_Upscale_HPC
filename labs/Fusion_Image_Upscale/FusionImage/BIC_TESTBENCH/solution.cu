// do not modify this file for histogram versions 0 and 1
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>

#include <chrono>

#include "kernel.cu"

int main(int argc, char* argv[])
{

    int scale = atoi(argv[1]); int block_dim = atoi(argv[2]);
    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char* h_img;                               //Original Small Input Image
    unsigned char* h_BIC;                                //Upscaled BICUBIC
    unsigned char* h_BIC_gray;                           //Upscaled BICUBIC

    //Device Array Pointers
    unsigned char* d_img;                               //Original Small Input Image
    RGBA_t* d_RGBA_img;                                 //Original Small Input Image w/ 32bit-pixel format
    
    unsigned char* d_naive_img_BIC;                      //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_basic_img_BIC;                      //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_shared_one_thread_out_BIC;          //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_shared_one_thread_in_BIC;           //Upscaled Nearest Neighbor Image w/ 32bit-pixel format

    RGBA_t* d_basic_img_BIC_rgba;                        //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t* d_shared_one_thread_out_BIC_rgba;            //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t* d_shared_one_thread_in_BIC_rgba;             //Upscaled Nearest Neighbor Image w/ 32bit-pixel format

    
    unsigned char* d_naive_gray_BIC;                     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_basic_gray_BIC;                     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_shared_one_thread_out_gray_BIC;     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_shared_one_thread_in_gray_BIC;      //Upscaled Greyscale Nearest Neighbor Image

   
    //Temporary Images for debug
    unsigned char* d_temp_output_img1;
    unsigned char* d_temp_output_img2;

    //Kernel Parameters
    bool RUBICING = true;
    bool firstImg = true;

    //Not sure these are needed will keep for now
    int window_size = 8;


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

        int max_image = 200;
        int current_img = 1;

        double processing_time = 0;
 
        //**************** Setup Kernel ****************//
        
//argv[1]
        strcpy(file_name, argv[3]); // copy string one into the result.
        strcat(file_name, "1.ppm"); // append string two to the result.
        
        printf(file_name);
        //sprintf(file_name, "./NV/image%d.ppm", current_img);
        //sprintf(file_name, "./LAD/LAD_%d.ppm", current_img);
        //sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);


        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        //******** Malloc Host Images ********//
        h_BIC = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
        h_BIC_gray = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
        //******** Malloc Host Images ********//

        //******** Malloc Device Images ********//

        //Original Image & RGBA Image
        if (cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_RGBA_img, width * height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Upscaled Images
        if (cudaMalloc((void**)&d_naive_img_BIC, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        
        if (cudaMalloc((void**)&d_basic_img_BIC, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_BIC, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_BIC, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        if (cudaMalloc((void**)&d_basic_img_BIC_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_BIC_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_BIC_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Gray Pictures
        if (cudaMalloc((void**)&d_naive_gray_BIC, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_basic_gray_BIC, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_gray_BIC, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_gray_BIC, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "BIC Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Grid_Input(((width - 1) / block_dim) + 1, ((height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block(block_dim, block_dim);

        dim3 Grid_16(((big_width - 1) / 16) + 1, ((big_height - 1) / 16) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block_16(16, 16);

        dim3 GRID_RGB_Convert(ceil((big_width * big_height) / 256.0));
        dim3 BLOCK_RGB_Convert(256);

        dim3 BiCubic_Block(4*scale, 4*scale);
        dim3 BiCubic_Grid(((big_width - 1) / BiCubic_Block.x) + 1, ((big_height - 1) / BiCubic_Block.y) + 1);
        int  BiCubic_Shared_Mem_Size = ((BiCubic_Block.y / scale) + 3) * ((BiCubic_Block.x / scale) + 3);

        cudaError_t error = cudaGetLastError();

        //**************** Setup Kernel ****************//

        //Variables for timing
        cudaEvent_t astartEvent, astopEvent;
        float aelapsedTime;
        cudaEventCreate(&astartEvent);
        cudaEventCreate(&astopEvent);

        //Load Input Image
        h_img = (unsigned char*)readPPM(file_name, &width, &height);

        //Copy Input Image to Device
        cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        //Convert original image to RGBA image
        rgbToRGBA_Kernel << < GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_RGBA_img, d_img, width * height);
        cudaDeviceSynchronize();

        //******************************* Run & Time Kernels ********************************//
        printf("BICUBIC Test\nBlock Dimensions, %d, Scale Factor, %d\nInput Image Dimensions, %d , %d\nOutput Image Dimensions, %d, %d\n", block_dim, scale, width, height, big_width, big_height);

        ////////////TIME BICUBIC WITH SERIAL CODE IMPLEMENTATION/////////////////////
        auto start = std::chrono::high_resolution_clock::now();

        bicubicInterpolation(h_BIC, big_width, big_height, h_img, width, height, scale);
        RGB2Greyscale(h_BIC_gray, h_BIC, big_width, big_height);

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - start;

        processing_time = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

        printf("BIC SERIAL CODE, %f, ms\n", processing_time/1000);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./BIC_TEST/BIC_SERIAL.ppm", (char*)h_BIC, big_width, big_height);
        writePPMGrey("./BIC_TEST/BIC_SERIAL_GRAY.ppm", (char*)h_BIC_gray, big_width, big_height);

        cudaDeviceSynchronize();

        ////////////TIME BICUBIC WITH NAIVE CUDA IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);
        if (block_dim > 16)
            bicubicInterpolationKernel <<< Grid_16, Block_16 >> > (d_naive_img_BIC, d_img, big_width, big_height, width, height, scale);
        else
            bicubicInterpolationKernel <<< Grid, Block >> > (d_naive_img_BIC, d_img, big_width, big_height, width, height, scale);

        cudaDeviceSynchronize();
        RGB2GreyscaleKernel << < Grid, Block >> > (d_naive_img_BIC, d_naive_gray_BIC, big_width, big_height);
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("BIC NAIVE CUDA, %f, ms\n", aelapsedTime);
        
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }


        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_BIC, d_naive_img_BIC, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_BIC_gray, d_naive_gray_BIC, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./BIC_TEST/BIC_NAIVE.ppm", (char*)h_BIC, big_width, big_height);
        writePPMGrey("./BIC_TEST/BIC_NAIVE_GRAY.ppm", (char*)h_BIC_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME BICUBIC WITH BASIC OPTIMIZED IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);
        if (block_dim > 16)
            bicubicInterpolation_GreyCon_Kernel_RGBA <<< Grid_16, Block_16 >> > (d_basic_img_BIC_rgba, d_basic_gray_BIC, d_RGBA_img, big_width, big_height, width, height, scale);
        else
            bicubicInterpolation_GreyCon_Kernel_RGBA << < Grid, Block >> > (d_basic_img_BIC_rgba, d_basic_gray_BIC, d_RGBA_img, big_width, big_height, width, height, scale);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("BIC BASIC OPTIMIZED, %f, ms\n", aelapsedTime);
        
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }


        rgbaToRGB_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_basic_img_BIC, d_basic_img_BIC_rgba, big_width * big_height);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_BIC, d_basic_img_BIC, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_BIC_gray, d_basic_gray_BIC, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./BIC_TEST/BIC_BASIC.ppm", (char*)h_BIC, big_width, big_height);
        writePPMGrey("./BIC_TEST/BIC_BASIC_GRAY.ppm", (char*)h_BIC_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME BICUBIC WITH SHARED MEM ONE PER INPUT IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA <<<BiCubic_Grid, BiCubic_Block, sizeof(RGBA_t)* BiCubic_Shared_Mem_Size>> > 
                                            (d_shared_one_thread_in_BIC_rgba, d_shared_one_thread_in_gray_BIC, d_RGBA_img, big_width, big_height, width, height, scale);
        
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("BIC SHARED MEM ONE THREAD PER INPUT, %f, ms\n", aelapsedTime);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }


        rgbaToRGB_Kernel << < GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_shared_one_thread_in_BIC, d_shared_one_thread_in_BIC_rgba, big_width * big_height);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_BIC, d_shared_one_thread_in_BIC, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_BIC_gray, d_shared_one_thread_in_gray_BIC, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./BIC_TEST/BIC_SHARED.ppm", (char*)h_BIC, big_width, big_height);
        writePPMGrey("./BIC_TEST/BIC_SHARED_GRAY.ppm", (char*)h_BIC_gray, big_width, big_height);
        cudaDeviceSynchronize();


        //************************* CHECK FOR CORRECTNESS *****************************//
        printf("\n\nCHECKING FOR CORRECTNESS\n");

        //Compare with Serial Image
        unsigned char* serial_check_img             = (unsigned char*)readPPM("./BIC_TEST/BIC_SERIAL.ppm", &big_width, &big_height);
        unsigned char* serial_check_img_gray        = (unsigned char*)readPPMGray("./BIC_TEST/BIC_SERIAL_GRAY.ppm", &big_width, &big_height);

        //TEST 1: NAIVE CUDA
        unsigned char* naive_check_img              = (unsigned char*)readPPM("./BIC_TEST/BIC_NAIVE.ppm", &big_width, &big_height);
        unsigned char* naive_check_img_gray         = (unsigned char*)readPPMGray("./BIC_TEST/BIC_NAIVE_GRAY.ppm", &big_width, &big_height);

        printf("CHECKING NAIVE CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, naive_check_img,        big_width, big_height);
        printf("CHECKING NAIVE CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, naive_check_img_gray,   big_width, big_height);

        //TEST 2: BASIC OPTIMIZED CUDA
        unsigned char* basic_check_img              = (unsigned char*)readPPM("./BIC_TEST/BIC_BASIC.ppm", &width, &height);
        unsigned char* basic_check_img_gray         = (unsigned char*)readPPMGray("./BIC_TEST/BIC_BASIC_GRAY.ppm", &width, &height);

        printf("CHECKING BASIC OPTIMIZED CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, basic_check_img, big_width, big_height);
        printf("CHECKING BASIC OPTIMIZED CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, basic_check_img_gray, big_width, big_height);

        //TEST 3: SHARED MEMORY ONE THREAD PER INPUT PIXEL CUDA
        unsigned char* shared_otpi_check_img        = (unsigned char*)readPPM("./BIC_TEST/BIC_SHARED.ppm", &width, &height);
        unsigned char* shared_otpi_check_img_gray   = (unsigned char*)readPPMGray("./BIC_TEST/BIC_SHARED_GRAY.ppm", &width, &height);

        printf("CHECKING SHARED MEM - COLORED,");
        Image_Compare(serial_check_img, shared_otpi_check_img, big_width, big_height);
        printf("CHECKING SHARED MEM - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, shared_otpi_check_img_gray, big_width, big_height);

    
        free(serial_check_img);         free(serial_check_img_gray);
        free(naive_check_img);          free(naive_check_img_gray);
        free(basic_check_img);          free(basic_check_img_gray);
        free(shared_otpi_check_img);    free(shared_otpi_check_img_gray);

        //************************* CLEAN UP *****************************//
        // 
        // TODO ACTUALLY FREE THE MEMORY LOL
        //Free Host Memory
        free(h_img);        free(h_BIC);     free(h_BIC_gray);

        //Free device Memory
        cudaFree(d_img);                            cudaFree(d_RGBA_img);
        cudaFree(d_naive_img_BIC);                   cudaFree(d_basic_img_BIC);
        cudaFree(d_shared_one_thread_out_BIC);       cudaFree(d_shared_one_thread_in_BIC);

        cudaFree(d_basic_img_BIC_rgba);              cudaFree(d_shared_one_thread_out_BIC_rgba);
        cudaFree(d_shared_one_thread_in_BIC_rgba);

        cudaFree(d_naive_gray_BIC);                  cudaFree(d_basic_gray_BIC);
        cudaFree(d_shared_one_thread_out_gray_BIC);  cudaFree(d_shared_one_thread_in_gray_BIC);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}