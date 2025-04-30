// Final Algorithm Set For Fusion Upscaling
// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

#include "util.cu"
#include "kernel.cu"

//#define STREAM_COUNT 2

int main(int argc, char* argv[])
{
    // ARGUMENTS (Example Below)
    // ./CUDA_STREAM_CODE_TEST_BENCH_Solution 2 ./NV/image "Fallout New Vegas" 1 10 2
    // EXECUTABLE, SCALE, FILE_FORMAT, GAME_NAME, START_FRAME, END_FRAME, STREAM NUMBERS
    //          0,     1,           2,         3,           4,         5,              6

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

    int STREAM_COUNT = atoi(argv[6]);
    
    memset(file_address, 0, sizeof(file_address));

    strcat(file_address, file_path); 
    sprintf(file_name, "%d.ppm", start_frame);
    strcat(file_address, file_name);

    //printf(file_address); printf("\n");

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
        printf("CUDA Code Test\n");
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

        printf("\n");
        printf("FINAL CUDA STREAM CODE, STREAMS, %d COMPUTE FPS, %f, Compute Execution Time for %d frames, %f, s\n", 
                                        STREAM_COUNT, total_fps_compute, total_frames, compute_execution_time_s);
        printf("FINAL CUDA STREAM CODE, STREAMS, %d EFFECTIVE FPS, %f, Total Execution Time for %d frames, %f, s\n", 
                                        STREAM_COUNT, total_fps_program, total_frames, program_execution_time_s);

        for (int s = 0; s < STREAM_COUNT; s++)
        {
            //SAVE FINAL FRAMES
            memset(file_address, 0, sizeof(file_address));
            strcpy(game_name, argv[3]);     //copy game name into the var

            strcat(file_address, "./MAIN_OUTPUT/STREAM_OUTPUT/FUSED_"); 
            sprintf(file_name, "Streams_%d_Scale_%d_Game_%s_StreamId_%d.ppm", STREAM_COUNT, scale, game_name, s);
            strcat(file_address, file_name);
            
            //printf("%s\n", file_address);
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

        //Load Final Image from the last stream.
        strcpy(game_name, argv[3]);     //copy game name into the var
        memset(file_address, 0, sizeof(file_address));
        strcat(file_address, "./MAIN_OUTPUT/STREAM_OUTPUT/FUSED_"); 
        sprintf(file_name, "Streams_%d_Scale_%d_Game_%s_StreamId_%d.ppm", STREAM_COUNT, scale, game_name, STREAM_COUNT-1);
        strcat(file_address, file_name);
        unsigned char* Streamed_Img = (unsigned char*)readPPM(file_address, &width, &height);

        //Compare Images with Serial Output
        Image_Compare(Serial_Img, Streamed_Img, 3, big_width, big_height);

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
