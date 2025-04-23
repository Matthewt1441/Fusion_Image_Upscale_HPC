#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>
#include "util.cu"

__global__ void nearestNeighborsKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;                    
    int Col = blockIdx.x * blockDim.x + threadIdx.x;                      

    int small_x = 0;    int small_y = 0;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        big_img_data[3 * (Row * big_width + Col) + 0] = img_data[3 * (small_y * width + small_x) + 0];
        big_img_data[3 * (Row * big_width + Col) + 1] = img_data[3 * (small_y * width + small_x) + 1];
        big_img_data[3 * (Row * big_width + Col) + 2] = img_data[3 * (small_y * width + small_x) + 2];
    }
}

__device__ float cubicInterpolateDevice(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice(float p[4][4], float y, float x)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice(p[0], x);
    arr[1] = cubicInterpolateDevice(p[1], x);
    arr[2] = cubicInterpolateDevice(p[2], x);
    arr[3] = cubicInterpolateDevice(p[3], x);
    return cubicInterpolateDevice(arr, y);
}

__global__ void bicubicInterpolationKernel(unsigned char* big_img_data, unsigned char* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int input_x = 0;
    int input_y = 0;

    int window_x = 0;
    int window_y = 0;
    
    int output_x = Col;
    int output_y = Row;


    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int sample_x = 0;
    int sample_y = 0;

    for (window_y = 0; window_y < 4; window_y++)
    {
        for (window_x = 0; window_x < 4; window_x++)
        {
            window_r[window_y][window_x] = 0;
            window_g[window_y][window_x] = 0;
            window_b[window_y][window_x] = 0;
        }
    }

    if(output_y < big_height &&  output_x < big_width)
    {
        //Calculate starting index for windows
        float interpolated_x = (float)(output_x / (scale * 1.0));
        float interpolated_y = (float)(output_y / (scale * 1.0));

        int input_block_start_idx_x = (output_x / scale);
        int input_block_start_idx_y = (output_y / scale);

        float dx = interpolated_x - input_block_start_idx_x;
        float dy = interpolated_y - input_block_start_idx_y;

        //We are within a block of the input image, therefore fill windows
        for(window_y = -1; window_y < 3; window_y++)
        {
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image index
                input_x = input_block_start_idx_x + window_x;
                input_y = input_block_start_idx_y + window_y;

                // Fill window with Nearest Neighbor edge behavior
                if(input_x < 0 || input_x >= width)
                {
                    // Find nearest in-bounds pixel
                    input_x = (input_x < 0) ? 0 : width - 1;
                }
                // Fill window with Nearest Neighbor edge behavior
                if(input_y < 0 || input_y >= height)
                {
                    // Find nearest in-bounds pixel
                    input_y = (input_y < 0) ? 0 : height - 1;
                }

                window_r[window_y + 1][window_x + 1] = (float)img_data[3 * (input_y * width + input_x) + 0];    //R
                window_g[window_y + 1][window_x + 1] = (float)img_data[3 * (input_y * width + input_x) + 1];    //G
                window_b[window_y + 1][window_x + 1] = (float)img_data[3 * (input_y * width + input_x) + 2];    //B
            }
        }

        float r = bicubicInterpolateDevice(window_r, dy, dx);
        float g = bicubicInterpolateDevice(window_g, dy, dx);
        float b = bicubicInterpolateDevice(window_b, dy, dx);

        big_img_data[3 * (output_y * big_width + output_x) + 0] = (unsigned char)r;
        big_img_data[3 * (output_y * big_width + output_x) + 1] = (unsigned char)g;
        big_img_data[3 * (output_y * big_width + output_x) + 2] = (unsigned char)b;
    }
}

__global__ void RGB2GreyscaleKernel(unsigned char* rgb_img, unsigned char* grey_img, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        int rgbidx = rgbidx = 3 * (Row * width + Col);
        grey_img[Row * width + Col] = 0.21f * rgb_img[rgbidx + 0] + 0.71f * rgb_img[rgbidx + 1] + 0.07f * rgb_img[rgbidx + 2];
    }
    
}

__global__ void ABS_Difference_Grey_Kernel(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    float img_1_signed = 0;
    float img_2_signed = 0;

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        img_1_signed = (float)img_1[Row * width + Col];
        img_2_signed = (float)img_2[Row * width + Col];

        diff_map[Row * width + Col] = (float)abs((img_1_signed - img_2_signed) / 255.0); //Normalize 
    }
}

__device__ float calculateSSIMDevice(float window1[8][8], float window2[8][8], int window_width, int window_height)
{
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    int size = window_height * window_width;
    int valid_count = 0;

    for (int i = 0; i < window_height; ++i) 
    {
        for (int j = 0; j < window_width; ++j)
        {
            if ((window1[i][j] >= 0) && (window2[i][j] >= 0))
            {
                sum1 += window1[i][j];
                sum2 += window2[i][j];
                sum1Sq += window1[i][j] * window1[i][j];
                sum2Sq += window2[i][j] * window2[i][j];
                sum12 += window1[i][j] * window2[i][j];
                valid_count++;
            }
        }
    }

    float mu1 = sum1 / valid_count;
    float mu2 = sum2 / valid_count;
    float sigma1Sq = (sum1Sq / valid_count) - (mu1 * mu1);
    float sigma2Sq = (sum2Sq / valid_count) - (mu2 * mu2);
    float sigma12 = (sum12 / valid_count) - (mu1 * mu2);

    // Stabilizing constants
    float C1 = 6.5025; // (K1*L)^2, where K1=0.01 and L=255
    float C2 = 58.5225; // (K2*L)^2, where K2=0.03 and L=255

    float ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2));
    return ssim;
}


__global__ void SSIM_Grey_Kernel(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the guassian option with an 11x11 window
    float window_img1[8][8] = { 0 };
    float window_img2[8][8] = { 0 };

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //For now, generate a smaller image.
    if (Row < height && Col < width)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                if (((Row + i) * width + (Col + j)) < (width * height))
                {
                    window_img1[i][j] = img_1[(Row + i) * width + (Col + j)];
                    window_img2[i][j] = img_2[(Row + i) * width + (Col + j)];
                }
                else
                {
                    window_img1[i][j] = -1;
                    window_img2[i][j] = -1;
                }
            }
        }

        ssim_map[Row * width + Col] = calculateSSIMDevice(window_img1, window_img2, 8, 8);
    }
}

__global__ void MapMulKernel(float* product_map, float* map_1, float* map_2, int width, int height)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int idx = Row * width + Col;
    
    if (Row < height && Col < width)
    {
        product_map[idx] = map_1[idx] * map_2[idx];
    }
}

__global__ void GuassianBlur_Map_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma)
{
    //Generate Normalized Guassian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float guassian_kernel[49] = { 0 };

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float my_PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

    if (Row < height && Col < width)
    {
        for (int y = 0; y < kernel_size; y++)
        {
            for (int x = 0; x < kernel_size; x++)
            {
                double exponent = -((x - kernel_center) * (x - kernel_center) - (y - kernel_center) * (y - kernel_center)) / (2 * sigma * sigma);
                guassian_kernel[y * kernel_size + x] = exp(exponent) / (2 * my_PI * sigma * sigma);
                sum += guassian_kernel[y * kernel_size + x];
            }
        }
        //Normalize
        //May not want to do this as edge cases will not utilize entire kernel.
        //Will try for now. It may be the right way to do it. I don't know for sure.
        for (int i = 0; i < kernel_size; i++) 
            for (int j = 0; j < kernel_size; j++) 
                guassian_kernel[i * kernel_size + j] /= sum;

        sum = 0.0;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int map_y = Row + i - radius; //
                int map_x = Col + j - radius;

                //If we are within the image
                if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                    sum += input_map[map_y * width + map_x] * guassian_kernel[i * kernel_size + j];
                }
            }
        }

        blur_map[Row * width + Col] = sum;
    }
}

__global__ void MapThreshold_Kernel(float* map, float threshold, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = Row * width + Col;

    if (Row < height && Col < width)
    {

        if (map[idx] > threshold)
        {
            map[idx] = 1.0;
        }
        else
        {
            map[idx] = 0.0;
        }
    }
}

__global__ void Image_Fusion_Kernel(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int map_idx = Row * width + Col;
    int img_idx = 3 * map_idx;

    if (Row < height && Col < width)
    {
        fused_img[img_idx + 0] = img_1[img_idx + 0] * weight_map[map_idx] + img_2[img_idx + 0] * (1.0 - weight_map[map_idx]);
        fused_img[img_idx + 1] = img_1[img_idx + 1] * weight_map[map_idx] + img_2[img_idx + 1] * (1.0 - weight_map[map_idx]);
        fused_img[img_idx + 2] = img_1[img_idx + 2] * weight_map[map_idx] + img_2[img_idx + 2] * (1.0 - weight_map[map_idx]);
    }
}