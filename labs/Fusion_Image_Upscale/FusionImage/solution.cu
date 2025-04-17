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
    unsigned char* h_nn;                                //Upscaled Nearest Neighbors
    unsigned char* h_nn_gray;                           //Upscaled Nearest Neighbors

    //Device Array Pointers
    unsigned char* d_img;                               //Original Small Input Image
    RGBA_t* d_RGBA_img;                                 //Original Small Input Image w/ 32bit-pixel format
    
    unsigned char* d_naive_img_nn;                      //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_basic_img_nn;                      //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_shared_one_thread_out_nn;          //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    unsigned char* d_shared_one_thread_in_nn;           //Upscaled Nearest Neighbor Image w/ 32bit-pixel format

    RGBA_t* d_basic_img_nn_rgba;                        //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t* d_shared_one_thread_out_nn_rgba;            //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t* d_shared_one_thread_in_nn_rgba;             //Upscaled Nearest Neighbor Image w/ 32bit-pixel format

    
    unsigned char* d_naive_gray_nn;                     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_basic_gray_nn;                     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_shared_one_thread_out_gray_nn;     //Upscaled Greyscale Nearest Neighbor Image
    unsigned char* d_shared_one_thread_in_gray_nn;      //Upscaled Greyscale Nearest Neighbor Image

   
    //Temporary Images for debug
    unsigned char* d_temp_output_img1;
    unsigned char* d_temp_output_img2;

    //Kernel Parameters
    bool RUNNING = true;
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
        char fps_str[50];
        char file_name[50];
        int count = 0;

        double frame_cap = 10;
        sprintf(fps_str, "FPS:%.*f", 3, 0.0);

        int max_image = 200;
        int current_img = 1;

        double processing_time = 0;
 
        //Read in first image initially to get input width and height.
        //sprintf(file_name, "./NV/image%d.ppm", current_img);
        //sprintf(file_name, "./LAD/LAD_%d.ppm", current_img);
        sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);
        h_img = (unsigned char*)readPPM(file_name, &width, &height);
        free(h_img);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;

        //******** Malloc Host Images ********//
        h_nn = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height * 3);
        h_nn_gray = (unsigned char*)malloc(sizeof(unsigned char) * big_width * big_height);
        //******** Malloc Host Images ********//

        //******** Malloc Device Images ********//

        //Original Image & RGBA Image
        if (cudaMalloc((void**)&d_img, width * height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_RGBA_img, width * height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Upscaled Images
        if (cudaMalloc((void**)&d_naive_img_nn, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        
        if (cudaMalloc((void**)&d_basic_img_nn, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_nn, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_nn, big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        if (cudaMalloc((void**)&d_basic_img_nn_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_nn_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_nn_rgba, big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
            fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

        //Gray Pictures
        if (cudaMalloc((void**)&d_naive_gray_nn, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_basic_gray_nn, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_out_gray_nn, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        if (cudaMalloc((void**)&d_shared_one_thread_in_gray_nn, big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
            fprintf(stderr, "NN Grey Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));


        //**************** Setup Kernel ****************//
        //sprintf(file_name, "./NV/image%d.ppm", current_img);
        //sprintf(file_name, "./LAD/LAD_%d.ppm", current_img);
        sprintf(file_name, "./LM_Frame/image%d.ppm", current_img);

        dim3 Grid(((big_width - 1) / block_dim) + 1, ((big_height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Grid_Input(((width - 1) / block_dim) + 1, ((height - 1) / block_dim) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 Block(block_dim, block_dim);

        dim3 GRID_RGB_Convert(ceil((big_width * big_height) / 256.0));
        dim3 BLOCK_RGB_Convert(256);

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
        printf("Nearest Neighbors Test\nBlock Dimensions, %d, Scale Factor, %d\nInput Image Dimensions, %d , %d\nOutput Image Dimensions, %d, %d\n", block_dim, scale, width, height, big_width, big_height);

        ////////////TIME NEAREST NEIGHBORS WITH SERIAL CODE IMPLEMENTATION/////////////////////
        auto start = std::chrono::high_resolution_clock::now();

        nearestNeighbors(h_nn_gray, h_nn, big_width, big_height, h_img, width, height, scale);
        //RGB2Greyscale(h_nn_gray, h_nn, big_width, big_height);

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - start;

        processing_time = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

        printf("NN SERIAL CODE, %f, ms\n", processing_time/1000);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./NN_TEST/NN_SERIAL.ppm", (char*)h_nn, big_width, big_height);
        writePPMGrey("./NN_TEST/NN_SERIAL_GRAY.ppm", (char*)h_nn_gray, big_width, big_height);

        cudaDeviceSynchronize();

        ////////////TIME NEAREST NEIGHBORS WITH NAIVE CUDA IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        nearestNeighborsKernel << < Grid, Block >> > (d_naive_img_nn, d_img, big_width, big_height, width, height, scale);
        RGB2GreyscaleKernel << < Grid, Block >> > (d_naive_img_nn, d_naive_gray_nn, big_width, big_height);
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("NN NAIVE CUDA, %f, ms\n", aelapsedTime);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_nn, d_naive_img_nn, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nn_gray, d_naive_gray_nn, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./NN_TEST/NN_NAIVE.ppm", (char*)h_nn, big_width, big_height);
        writePPMGrey("./NN_TEST/NN_NAIVE_GRAY.ppm", (char*)h_nn_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME NEAREST NEIGHBORS WITH BASIC OPTIMIZED IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        nearestNeighbors_GreyCon_Kernel_RGBA << < Grid, Block >> > (d_basic_img_nn_rgba, d_basic_gray_nn, d_RGBA_img, big_width, big_height, width, height, scale);
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("NN BASIC OPTIMIZED, %f, ms\n", aelapsedTime);
        
        rgbaToRGB_Kernel <<< GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_basic_img_nn, d_basic_img_nn_rgba, big_width * big_height);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_nn, d_basic_img_nn, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nn_gray, d_basic_gray_nn, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./NN_TEST/NN_BASIC.ppm", (char*)h_nn, big_width, big_height);
        writePPMGrey("./NN_TEST/NN_BASIC_GRAY.ppm", (char*)h_nn_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME NEAREST NEIGHBORS WITH SHARED MEM ONE PER INPUT IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        nearestNeighbors_shared_memory_Kernel <<< Grid_Input, Block, sizeof(RGBA_t)* block_dim * block_dim>> > 
                                            (d_shared_one_thread_in_nn_rgba, d_shared_one_thread_in_gray_nn, d_RGBA_img, big_width, big_height, width, height, scale);
        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("NN SHARED MEM ONE THREAD PER INPUT, %f, ms\n", aelapsedTime);

        rgbaToRGB_Kernel << < GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_shared_one_thread_in_nn, d_shared_one_thread_in_nn_rgba, big_width * big_height);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_nn, d_shared_one_thread_in_nn, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nn_gray, d_shared_one_thread_in_gray_nn, sizeof(unsigned char) * big_width * big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./NN_TEST/NN_SHARED_ONE_THREAD_INPUT.ppm", (char*)h_nn, big_width, big_height);
        writePPMGrey("./NN_TEST/NN_SHARED_ONE_THREAD_INPUT_GRAY.ppm", (char*)h_nn_gray, big_width, big_height);
        cudaDeviceSynchronize();

        ////////////TIME NEAREST NEIGHBORS WITH SHARED MEM ONE PER INPUT IMPLEMENTATION/////////////////////
        cudaEventRecord(astartEvent, 0);

        nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel << < Grid, Block, sizeof(RGBA_t)* block_dim* block_dim / scale >> > 
                                (d_shared_one_thread_out_nn_rgba, d_shared_one_thread_out_gray_nn, d_RGBA_img, big_width, big_height, width, height, scale);

        cudaDeviceSynchronize();
        cudaEventRecord(astopEvent, 0);
        cudaEventSynchronize(astopEvent);
        cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
        printf("NN SHARED MEM ONE THREAD PER OUTPUT, %f, ms\n", aelapsedTime);

        rgbaToRGB_Kernel << < GRID_RGB_Convert, BLOCK_RGB_Convert >> > (d_shared_one_thread_out_nn, d_shared_one_thread_out_nn_rgba, big_width* big_height);

        cudaDeviceSynchronize();

        //COPY OVER IMAGE DATA
        cudaMemcpy(h_nn, d_shared_one_thread_out_nn, sizeof(unsigned char)* big_width* big_height * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_nn_gray, d_shared_one_thread_out_gray_nn, sizeof(unsigned char)* big_width* big_height, cudaMemcpyDeviceToHost);

        //WRITE THE PICTURES FOR COMPARISON
        writePPM("./NN_TEST/NN_SHARED_ONE_THREAD_OUTPUT.ppm", (char*)h_nn, big_width, big_height);
        writePPMGrey("./NN_TEST/NN_SHARED_ONE_THREAD_OUTPUT_GRAY.ppm", (char*)h_nn_gray, big_width, big_height);
        cudaDeviceSynchronize();


        //************************* CHECK FOR CORRECTNESS *****************************//
        printf("\n\nCHECKING FOR CORRECTNESS\n");

        //Compare with Serial Image
        unsigned char* serial_check_img             = (unsigned char*)readPPM("./NN_TEST/NN_SERIAL.ppm", &big_width, &big_height);
        unsigned char* serial_check_img_gray        = (unsigned char*)readPPMGray("./NN_TEST/NN_SERIAL_GRAY.ppm", &big_width, &big_height);

        //TEST 1: NAIVE CUDA
        unsigned char* naive_check_img              = (unsigned char*)readPPM("./NN_TEST/NN_NAIVE.ppm", &big_width, &big_height);
        unsigned char* naive_check_img_gray         = (unsigned char*)readPPMGray("./NN_TEST/NN_NAIVE_GRAY.ppm", &big_width, &big_height);

        printf("CHECKING NAIVE CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, naive_check_img,        big_width, big_height);
        printf("CHECKING NAIVE CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, naive_check_img_gray,   big_width, big_height);

        //free(naive_check_img);
        //free(naive_check_img_gray);

        //TEST 2: BASIC OPTIMIZED CUDA
        unsigned char* basic_check_img              = (unsigned char*)readPPM("./NN_TEST/NN_BASIC.ppm", &width, &height);
        unsigned char* basic_check_img_gray         = (unsigned char*)readPPMGray("./NN_TEST/NN_BASIC_GRAY.ppm", &width, &height);

        printf("CHECKING BASIC OPTIMIZED CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, basic_check_img, big_width, big_height);
        printf("CHECKING BASIC OPTIMIZED CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, basic_check_img_gray, big_width, big_height);

        //TEST 3: SHARED MEMORY ONE THREAD PER INPUT PIXEL CUDA
        unsigned char* shared_otpi_check_img        = (unsigned char*)readPPM("./NN_TEST/NN_SHARED_ONE_THREAD_INPUT.ppm", &width, &height);
        unsigned char* shared_otpi_check_img_gray   = (unsigned char*)readPPMGray("./NN_TEST/NN_SHARED_ONE_THREAD_INPUT_GRAY.ppm", &width, &height);

        printf("CHECKING SHARED MEM ONE THREAD PER INPUT CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, shared_otpi_check_img, big_width, big_height);
        printf("CHECKING SHARED MEM ONE THREAD PER INPUT CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, shared_otpi_check_img_gray, big_width, big_height);

        //TEST 4: SHARED MEMORY ONE THREAD PER OUTPUT PIXEL CUDA
        unsigned char* shared_otpo_check_img        = (unsigned char*)readPPM("./NN_TEST/NN_SHARED_ONE_THREAD_OUTPUT.ppm", &width, &height);
        unsigned char* shared_otpo_check_img_gray   = (unsigned char*)readPPMGray("./NN_TEST/NN_SHARED_ONE_THREAD_OUTPUT_GRAY.ppm", &width, &height);

        printf("CHECKING SHARED MEM ONE THREAD PER OUTPUT CUDA KERNEL - COLORED,");
        Image_Compare(serial_check_img, shared_otpo_check_img, big_width, big_height);
        printf("CHECKING SHARED MEM ONE THREAD PER OUTPUT CUDA KERNEL - GRAYSCALE,");
        Grey_Image_Compare(serial_check_img_gray, shared_otpo_check_img_gray, big_width, big_height);

    
        free(serial_check_img);         free(serial_check_img_gray);
        free(naive_check_img);          free(naive_check_img_gray);
        free(basic_check_img);          free(basic_check_img_gray);
        free(shared_otpi_check_img);    free(shared_otpi_check_img_gray);
        free(shared_otpo_check_img);    free(shared_otpo_check_img_gray);

        //************************* CLEAN UP *****************************//
        // 
        // TODO ACTUALLY FREE THE MEMORY LOL
        //Free Host Memory
        free(h_img);        free(h_nn);     free(h_nn_gray);

        //Free device Memory
        cudaFree(d_img);                            cudaFree(d_RGBA_img);
        cudaFree(d_naive_img_nn);                   cudaFree(d_basic_img_nn);
        cudaFree(d_shared_one_thread_out_nn);       cudaFree(d_shared_one_thread_in_nn);

        cudaFree(d_basic_img_nn_rgba);              cudaFree(d_shared_one_thread_out_nn_rgba);
        cudaFree(d_shared_one_thread_in_nn_rgba);

        cudaFree(d_naive_gray_nn);                  cudaFree(d_basic_gray_nn);
        cudaFree(d_shared_one_thread_out_gray_nn);  cudaFree(d_shared_one_thread_in_gray_nn);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    cudaDeviceReset();
    return 0;
}