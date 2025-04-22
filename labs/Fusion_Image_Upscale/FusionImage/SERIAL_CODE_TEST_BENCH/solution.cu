// call each kernel implemented in the kernel.cu
// generates timing info
// tests for functional verification

#include <cuda_runtime.h>
#include <stdlib.h>

#include <chrono>

#include "util.cu"
#include "serial_code.cu"

int main(int argc, char* argv[])
{
    // ARGUMENTS (Example Below)
    // ./SERIAL_CODE_TEST_BENCH_Solution 2 ./NV/image "Fallout New Vegas" 1 10
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

        //Timer
        auto program_start = std::chrono::high_resolution_clock::now();

        //******************************* Run & Time CODE ********************************//
        printf("Serial Code Test\n");
        printf("Scale Factor, %d, Game, %s, Start Frame, %d, End Frame, %d\n", scale, game_name, start_frame, end_frame);
        printf("Input Image Dimensions, %d , %d, Output Image Dimensions, %d, %d\n", width, height, big_width, big_height);

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
        writePPM("./SERIAL_OUPUT/FUSED.ppm", (char*)h_big_img_fused, big_width, big_height);


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

    cudaDeviceReset();
    return 0;
}