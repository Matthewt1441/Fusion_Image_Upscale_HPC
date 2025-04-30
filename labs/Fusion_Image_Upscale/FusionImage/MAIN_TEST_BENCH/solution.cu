// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>

#include <chrono>

#include "util.cu"
#include "serial_code.cu"
#include "kernel.cu"


char file_path[50];
char file_name[50];
char file_address[50];

char game_name[50];

int scale; 

int start_frame;
int end_frame;
int STREAM_COUNT;


int serial_version()
{
    int width;
    int height;
    int current_frame = start_frame;

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

    //Lets start off with timing one image
    try
    {
        //***** Temp *****//
        double processing_time = 0;
 
        //**************** Setup Kernel ****************//
        h_img = (unsigned char*)readPPM(file_address, &width, &height);
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
        h_big_img_fused                 = (unsigned char*)malloc(sizeof(unsigned char) * big_pixel_count * 3);
        h_diff_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_ssim_map                      = (float*)malloc(sizeof(float) * big_pixel_count);
        h_artifact_map                  = (float*)malloc(sizeof(float) * big_pixel_count);
        h_blurred_artifact_map          = (float*)malloc(sizeof(float) * big_pixel_count);
        //******** Malloc Host Images ********//

        float execution_time_ms = 0, execution_time_s;
        float current_fps = 0;
        float average_execution_time = 0;
        float average_fps = 0;

        //******************************* Run & Time CODE ********************************//
        printf("SERIAL Code Test\n");
        printf("Scale Factor, %d, Game, %s, Start Frame, %d, End Frame, %d\n", scale, game_name, start_frame, end_frame);
        printf("Input Image Dimensions, %d , %d, Output Image Dimensions, %d, %d\n", width, height, big_width, big_height);

        //Timer
        auto program_start = std::chrono::high_resolution_clock::now();

        while (current_frame <= end_frame)
        {
            //PHASE 0 : Load Input Image
            memset(file_address, 0, sizeof(file_address));

            strcat(file_address, file_path); 
            sprintf(file_name, "%d.ppm", current_frame);
            strcat(file_address, file_name);

            //printf(file_address); printf("\n");

            h_img = (unsigned char*)readPPM(file_address, &width, &height);

            //Timer
            auto start = std::chrono::high_resolution_clock::now();

            //PHASE 1 : IMAGE UPSCALE WITH NEAREST NEIGHBORS & BICUBIC
            nearestNeighbors(h_big_img_nn, big_width, big_height, h_img, width, height, scale);
            bicubicInterpolation(h_big_img_bic, big_width, big_height, h_img, width, height, scale);

            //PHASE 2 : IMAGE GREY SCALE CONVERSION
            RGB2Greyscale(h_big_img_nn_grey, h_big_img_nn, big_width, big_height);
            RGB2Greyscale(h_big_img_bic_grey, h_big_img_bic, big_width, big_height);
            
            //PHASE 3 : ARTIFACT MAP CREATION
            ABS_Difference_Grey(h_diff_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
            SSIM_Grey(h_ssim_map, h_big_img_nn_grey, h_big_img_bic_grey, big_width, big_height);
            MapMul(h_artifact_map, h_diff_map, h_ssim_map, big_width, big_height);

            //PHASE 4 : MAP BLUR
            GaussianBlur_Map(h_blurred_artifact_map, h_artifact_map, big_width, big_height, 3, 1.5);
            MapThreshold(h_blurred_artifact_map, 0.05, big_width, big_height);

            //PHASE 5 : IMAGE FUSION
            Image_Fusion(h_big_img_fused, h_big_img_nn, h_big_img_bic, h_blurred_artifact_map, big_width, big_height);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = end - start;
            processing_time = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
            
            execution_time_ms = processing_time / 1000;
            execution_time_s = execution_time_ms / 1000;
            current_fps = 1/(execution_time_s);
            average_execution_time += execution_time_s;
            average_fps = 1/(average_execution_time / (current_frame - start_frame + 1));

            printf("SERIAL CODE: FRAME, %d, TIME, %f, ms, CURRENT FPS, %f, AVERAGE FPS, %f\n", current_frame, execution_time_ms, current_fps, average_fps);
            current_frame++;
        }

        auto program_end = std::chrono::high_resolution_clock::now();
        auto dur = program_end - program_start;
        processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        processing_time /= 1000; // ms / 1000 -> s

        //************************* CLEAN UP *****************************//
        printf("FINAL SERIAL CODE FPS, %f, Total Execution Time for %d frames, %f, s\n", average_fps, end_frame - start_frame + 1, processing_time);
        writePPM("./MAIN_OUTPUT/SERIAL_OUTPUT/FUSED.ppm", (char*)h_big_img_fused, big_width, big_height);


        //Free Host Memory
        free(h_img);                
        free(h_big_img_nn);         free(h_big_img_bic);   
        free(h_big_img_nn_grey);    free(h_big_img_bic_grey);
        free(h_diff_map);           free(h_ssim_map);
        free(h_artifact_map);       free(h_blurred_artifact_map);

        free(h_big_img_fused);
    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    catch (char const* err)
    {
        printf(err); printf("\n");
        return 1;
    }

    return 0;
}

int optimized_version()
{
    int current_frame = start_frame;
    
    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //Host Array Pointers, these should always be unsigned char
    unsigned char*  h_img;                              //Original Small Input Image
    unsigned char*  h_big_img_fused;                    //Upscaled Fused Image

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

        //Setup Gaussian Blur based on passed in Args
        int GAUSS_Ksize = 7;
        float GAUSS_Sigma = 1.5;
        dim3 h_Gauss_Block(256, 1);
        dim3 h_Gauss_Grid(((big_width - 1) / h_Gauss_Block.x) + 1, ((big_height - 1) / h_Gauss_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 v_Gauss_Block(8, 32);
        dim3 v_Gauss_Grid(((big_width - 1) / v_Gauss_Block.x) + 1, ((big_height - 1) / v_Gauss_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double

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
        printf("OPTIMIZED CUDA Code Test\n");
        printf("Scale Factor, %d, Game, %s, Start Frame, %d, End Frame, %d\n", scale, game_name, start_frame, end_frame);
        printf("Input Image Dimensions, %d , %d, Output Image Dimensions, %d, %d\n", width, height, big_width, big_height);

        cudaEventRecord(total_Time_Start, 0);           //BEGIN TOTAL EXECUTION TIME

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
            cudaMemcpy(d_img, h_img, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            //PHASE 1 : Image Pre-Processing
            cudaEventRecord(compute_Frame_Start, 0);    //BEGIN COMPUTATION FRAME TIMING
                                                        //Convert original image to RGBA image
            rgbToRGBA_Kernel << < RGB_Grid, RGB_Block, rgbToRGBA_Shared_Mem_Size >> > (d_RGBA_img, d_img, width * height);

            //PHASE 2 : Image Scaling
            nearestNeighbors_GreyCon_Kernel_RGBA                    <<< NN_Grid, NN_Block >>>                                   
                                (d_big_img_nn, d_big_img_nn_grey, d_RGBA_img, big_width, big_height, width, height, scale);
            bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA  <<< BiCubic_Grid, BiCubic_Block, BiCubic_Shared_Mem_Size >>>
                                (d_big_img_bic, d_big_img_bic_grey, d_RGBA_img, big_width, big_height, width, height, scale);

            //PHASE 3 : Image Artifact Detection
            Artifact_Shared_Memory_Kernel                           <<< Arti_Grid, Arti_Block, Arti_Shared_Mem_Size >>>         
                                (d_big_artifact_map, d_big_img_nn_grey, d_big_img_bic_grey, big_width, big_height);


            //PHASE 4 : Artifact Map Post Processing
            horizontalGaussianBlurConvolve  <<< h_Gauss_Grid, h_Gauss_Block, sizeof(float) * (h_Gauss_Block.x + GAUSS_Ksize - 1) * h_Gauss_Block.y >>>
                                (d_big_blurred_artifact_map_inter, d_big_artifact_map, big_width, big_height, GAUSS_Ksize);


            verticalGaussianBlurConvolve    <<< v_Gauss_Grid, v_Gauss_Block, sizeof(float) * (v_Gauss_Block.y + GAUSS_Ksize - 1) * v_Gauss_Block.x >>>
                                (d_big_blurred_artifact_map, d_big_blurred_artifact_map_inter, big_width, big_height, 0.05, GAUSS_Ksize);
            

            //PHASE 5 : Image Fusion
            Image_Fusion_Kernel_RGBA <<< RGB_Grid, RGB_Block >>>            
                                (d_big_rgba_img_fused       , d_big_img_nn, d_big_img_bic   , d_big_blurred_artifact_map, big_width, big_height);


            //PHASE 6 : Image Post Processing -> Convert Into Original Data Type
            rgbaToRGB_Kernel <<< RGB_Grid, RGB_Block, rgbaToRGB_Shared_Mem_Size>>> (d_big_img_fused, d_big_rgba_img_fused, big_width * big_height);
            cudaEventRecord(compute_Frame_End, 0);
            cudaEventSynchronize(compute_Frame_End);
            
            //Send Device Images to Host
            cudaDeviceSynchronize();
            cudaMemcpy(h_big_img_fused, d_big_img_fused, sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost);
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

        printf("FINAL CUDA OPTIMIZED CODE COMPUTE FPS, %f, Total Execution Time for %d frames, %f, s\n", compute_fps, end_frame - start_frame + 1, execution_time_s);
        printf("FINAL CUDA OPTIMIZED CODE FPS, %f, Total Execution Time for %d frames, %f, s\n", average_fps, end_frame - start_frame + 1, execution_time_s);

        memset(file_address, 0, sizeof(file_address));

        strcat(file_address, "./MAIN_OUTPUT/OPTIMIZED_OUTPUT/FUSED_"); 

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

int streamed_version()
{
    int current_frame = start_frame;

    int width;
    int height;

    int big_width;
    int big_height;
    int big_pixel_count;

    //STREAM 1 POINTERS
    //Host Array Pointers, these should always be unsigned char
    unsigned char*      h_img[STREAM_COUNT];                            //Original Small Input Image
    unsigned char*      h_big_img_fused[STREAM_COUNT];                  //Upscaled Fused Image

    //Device Array Pointers
    unsigned char*      d_img[STREAM_COUNT];                            //Original Small Input Image
    RGBA_t*             d_RGBA_img[STREAM_COUNT];                       //Original Small Input Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_nn[STREAM_COUNT];                     //Upscaled Nearest Neighbor Image w/ 32bit-pixel format
    RGBA_t*             d_big_img_bic[STREAM_COUNT];                    //Upscaled Bicubic Image w/ 32bit-pixel format
    unsigned char*      d_big_img_nn_grey[STREAM_COUNT];                //Upscaled Greyscale Nearest Neighbor Image
    unsigned char*      d_big_img_bic_grey[STREAM_COUNT];               //Upscaled Greyscale Bicubic Image
    float*              d_big_artifact_map[STREAM_COUNT];               //Upscaled Artifact Map for image fusion
    float*              d_big_blurred_artifact_map[STREAM_COUNT];       //Upscaled Blurred Artifact Map for image fusion
    float*              d_big_blurred_artifact_map_inter[STREAM_COUNT]; //Upscaled Blurred Artifact Map for image fusion
    RGBA_t*             d_big_rgba_img_fused[STREAM_COUNT];             //Upscaled Fused Image w/ 32bit-pixel format
    unsigned char*      d_big_img_fused[STREAM_COUNT];                  //Upscaled Fused Image  

    cudaStream_t stream[STREAM_COUNT];

    //Timer
    auto program_start = std::chrono::high_resolution_clock::now();

    try
    {
        //Check that CUDA-capable GPU is installed
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			exit(-1);
        }

        float processing_time = 0;
 
        //**************** Setup Kernel ****************//

        unsigned char* h_img_dim = (unsigned char*)readPPM(file_address, &width, &height);
        free(h_img_dim);

        //Define big image width and height
        big_width = width * scale; big_height = height * scale;
        big_pixel_count = big_width * big_height;
        int pixel_count = width * height;

        for (int s = 0; s < STREAM_COUNT; s++)
        {
            cudaStreamCreate(&stream[s]);

            //******** Malloc Host Images ********//
            cudaHostAlloc((void**) &h_img[s], sizeof(unsigned char) * pixel_count * 3, cudaHostAllocDefault);
            cudaHostAlloc((void**) &h_big_img_fused[s], sizeof(unsigned char) * big_pixel_count * 3 , cudaHostAllocDefault);
            //******** Malloc Host Images ********//
        
            //******** Malloc Device Images ********//
            //Original Image & RGBA Image
            if (cudaMalloc((void**)&d_img[s], width * height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_RGBA_img[s], width * height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "RGBA Original Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Upscaled Images
            if (cudaMalloc((void**)&d_big_img_nn[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "NN Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_img_bic[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "BIC Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Grey Versions for Upscaled Images
            if (cudaMalloc((void**)&d_big_img_nn_grey[s], big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "NN Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_img_bic_grey[s], big_width * big_height * sizeof(unsigned char)) != cudaSuccess)
                fprintf(stderr, "BIC Grey Big Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Maps for Fusion
            if (cudaMalloc((void**)&d_big_artifact_map[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_blurred_artifact_map_inter[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Intermediate Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_blurred_artifact_map[s], big_width * big_height * sizeof(float)) != cudaSuccess)
                fprintf(stderr, "Blured Artifact Map Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));

            //Final Image and RGBA Image
            if (cudaMalloc((void**)&d_big_img_fused[s], big_width * big_height * sizeof(unsigned char) * 3) != cudaSuccess)
                fprintf(stderr, "Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
            if (cudaMalloc((void**)&d_big_rgba_img_fused[s], big_width * big_height * sizeof(RGBA_t)) != cudaSuccess)
                fprintf(stderr, "RGBA Fused Image Failed to Malloc: %s\n", cudaGetErrorString(cudaStatus));
        }

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

        //Setup Gaussian Blur based on passed in Args
        int GAUSS_Ksize = 7;
        float GAUSS_Sigma = 1.5;
        dim3 h_Gauss_Block(256, 1);
        dim3 h_Gauss_Grid(((big_width - 1) / h_Gauss_Block.x) + 1, ((big_height - 1) / h_Gauss_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        dim3 v_Gauss_Block(8, 32);
        dim3 v_Gauss_Grid(((big_width - 1) / v_Gauss_Block.x) + 1, ((big_height - 1) / v_Gauss_Block.y) + 1);     //Calculate the number of blocks needed for the dimension. 1.0 * Forces Double
        int  h_Gauss_Mem = sizeof(float) * (h_Gauss_Block.x + GAUSS_Ksize - 1) * h_Gauss_Block.y;
        int  v_Gauss_Mem = sizeof(float) * (v_Gauss_Block.y + GAUSS_Ksize - 1) * v_Gauss_Block.x;

        //**************** Setup Kernel ****************//

        //******************************* Run & Time CODE ********************************//
        printf("STREAMED CUDA Code Test\n");
        printf("Scale Factor, %d, Game, %s, Start Frame, %d, End Frame, %d\n", scale, game_name, start_frame, end_frame);
        printf("Input Image Dimensions, %d , %d, Output Image Dimensions, %d, %d\n", width, height, big_width, big_height);

        auto compute_start = std::chrono::high_resolution_clock::now();

        while (current_frame <= end_frame)
        {
            for (int s = 0; s < STREAM_COUNT; s++)
            {
                //PHASE 0 : Load Input Image
                memset(file_address, 0, sizeof(file_address));

                strcat(file_address, file_path); 
                sprintf(file_name, "%d.ppm", current_frame + s);
                strcat(file_address, file_name);

                h_img[s] = (unsigned char*)readPPM(file_address, &width, &height);

                //Copy Input Image to Device
                cudaMemcpyAsync(d_img[s], h_img[s], sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice, stream[s]);

                //PHASE 1 : Image Pre-Processing : Convert original image to RGBA image
                rgbToRGBA_Kernel << < RGB_Grid, RGB_Block, rgbToRGBA_Shared_Mem_Size, stream[s] >> > (d_RGBA_img[s], d_img[s], width * height);

                //PHASE 2 : Image Scaling
                nearestNeighbors_GreyCon_Kernel_RGBA <<< NN_Grid, NN_Block, 0, stream[s] >>>                                   
                                    (d_big_img_nn[s], d_big_img_nn_grey[s], d_RGBA_img[s], big_width, big_height, width, height, scale);

                bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA  <<< BiCubic_Grid, BiCubic_Block, BiCubic_Shared_Mem_Size, stream[s] >>>
                                    (d_big_img_bic[s], d_big_img_bic_grey[s], d_RGBA_img[s], big_width, big_height, width, height, scale);

                //PHASE 3 : Image Artifact Detection
                Artifact_Shared_Memory_Kernel<<< Arti_Grid, Arti_Block, Arti_Shared_Mem_Size, stream[s] >>>         
                                    (d_big_artifact_map[s], d_big_img_nn_grey[s], d_big_img_bic_grey[s], big_width, big_height);

                //PHASE 4 : Artifact Map Post Processing
                horizontalGaussianBlurConvolve  <<< h_Gauss_Grid, h_Gauss_Block, h_Gauss_Mem, stream[s] >>>
                                    (d_big_blurred_artifact_map_inter[s], d_big_artifact_map[s], big_width, big_height, GAUSS_Ksize);

                verticalGaussianBlurConvolve    <<< v_Gauss_Grid, v_Gauss_Block, v_Gauss_Mem, stream[s] >>>
                                    (d_big_blurred_artifact_map[s], d_big_blurred_artifact_map_inter[s], big_width, big_height, 0.05, GAUSS_Ksize);

                //PHASE 5 : Image Fusion
                Image_Fusion_Kernel_RGBA <<< RGB_Grid, RGB_Block, 0, stream[s] >>>            
                                    (d_big_rgba_img_fused[s], d_big_img_nn[s], d_big_img_bic[s], d_big_blurred_artifact_map[s], big_width, big_height);

                //PHASE 6 : Image Post Processing -> Convert Into Original Data Type
                rgbaToRGB_Kernel <<< RGB_Grid, RGB_Block, rgbaToRGB_Shared_Mem_Size, stream[s]>>> 
                                    (d_big_img_fused[s], d_big_rgba_img_fused[s], big_width * big_height);

                //Send Device Images to Host
                cudaMemcpyAsync(h_big_img_fused[s], d_big_img_fused[s], sizeof(unsigned char) * big_width * big_height * 3, cudaMemcpyDeviceToHost, stream[s]);
    
            }
            current_frame += STREAM_COUNT;
        }

        auto    program_end = std::chrono::high_resolution_clock::now();
        
        auto    compute_dur = program_end - compute_start;
        auto    program_dur = program_end - program_start;

        double  compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(compute_dur).count();
        double  program_time = std::chrono::duration_cast<std::chrono::milliseconds>(program_dur).count();

        double  compute_execution_time_s = compute_time / 1000;
        double  program_execution_time_s = program_time / 1000;

        int     total_frames    = current_frame - start_frame;

        double  total_fps_compute   = total_frames / compute_execution_time_s;
        double  total_fps_program   = total_frames / program_execution_time_s;

        printf("FINAL CUDA STREAM CODE, STREAMS, %d COMPUTE FPS, %f, Compute Execution Time for %d frames, %f, s\n", 
                                        STREAM_COUNT, total_fps_compute, total_frames, compute_execution_time_s);
        printf("FINAL CUDA STREAM CODE, STREAMS, %d EFFECTIVE FPS, %f, Total Execution Time for %d frames, %f, s\n", 
                                        STREAM_COUNT, total_fps_program, total_frames, program_execution_time_s);

        for (int s = 0; s < STREAM_COUNT; s++)
        {
            //SAVE FINAL FRAMES
            memset(file_address, 0, sizeof(file_address));

            strcat(file_address, "./MAIN_OUTPUT/STREAM_OUTPUT/FUSED_"); 
            sprintf(file_name, "Streams_%d_Scale_%d_Stream_%d.ppm", STREAM_COUNT, scale, s);
            strcat(file_address, file_name);
            
            writePPM(file_address, (char*)h_big_img_fused[s], big_width, big_height);
            
            //************************* CLEAN UP *****************************//
            cudaStreamDestroy(stream[s]);

            //Free Host Memory
            cudaFreeHost(h_img[s]);                          cudaFreeHost(h_big_img_fused[s]);

            //Free device Memory
            cudaFree(d_img[s]);                              cudaFree(d_RGBA_img[s]);
            cudaFree(d_big_img_nn[s]);                       cudaFree(d_big_img_bic[s]);
            cudaFree(d_big_img_nn_grey[s]);                  cudaFree(d_big_img_bic_grey[s]);
            cudaFree(d_big_artifact_map[s]);                 cudaFree(d_big_blurred_artifact_map[s]);
            cudaFree(d_big_blurred_artifact_map_inter[s]);
            cudaFree(d_big_rgba_img_fused[s]);               cudaFree(d_big_img_fused[s]);
        }

        unsigned char* Serial_Img = (unsigned char*)readPPM("./MAIN_OUTPUT/SERIAL_OUTPUT/FUSED.ppm", &width, &height);

        //SAVE FINAL FRAMES
        memset(file_address, 0, sizeof(file_address));

        strcat(file_address, "./MAIN_OUTPUT/STREAM_OUTPUT/FUSED_"); 
            sprintf(file_name, "Streams_%d_Scale_%d_Stream_%d.ppm", STREAM_COUNT, scale, STREAM_COUNT-1);
        strcat(file_address, file_name);

        unsigned char* Streamed_Img = (unsigned char*)readPPM(file_address, &width, &height);
        printf("Hi\n");
        Image_Compare(Serial_Img, Streamed_Img, 3, width, height);
        printf("Hi2\n");


        free(Serial_Img);
        free(Streamed_Img);

    }

    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }
    
    catch (const char* err)
    {
        printf("%s\n", err);
        cudaDeviceReset();
        return 1;
    }
    cudaDeviceReset();
    return 0;
    
}

int main(int argc, char* argv[])
{
    // ARGUMENTS (Example Below)
    // ./SERIAL_CODE_TEST_BENCH_Solution 2 ./NV/image "Fallout New Vegas" 1 10
    // EXECUTABLE, SCALE, FILE_FORMAT, GAME_NAME, START_FRAME, END_FRAME, STREAM_COUNT
    //          0,     1,           2,         3,           4,         5,            6

    scale = atoi(argv[1]); 
            
    strcpy(file_path, argv[2]);     //copy file path into the var 
    strcpy(game_name, argv[3]);     //copy game name into the var

    start_frame = atoi(argv[4]);
    end_frame   = atoi(argv[5]);

    STREAM_COUNT = atoi(argv[6]);
    
    memset(file_address, 0, sizeof(file_address));

    strcat(file_address, file_path); 
    sprintf(file_name, "%d.ppm", start_frame);
    strcat(file_address, file_name);

    printf(file_address); printf("\n");

    serial_version();
    optimized_version();
    streamed_version();


    return 0;
}