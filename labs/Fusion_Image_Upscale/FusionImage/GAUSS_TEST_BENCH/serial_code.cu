#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

void RGB2Greyscale(unsigned char* grey_img, unsigned char* rgb_img, int width, int height)
{
    int rgbidx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            rgbidx = 3 * (y * width + x);
            grey_img[y * width + x] = 0.21f * rgb_img[rgbidx + 0] + 0.71f * rgb_img[rgbidx + 1] + 0.07f * rgb_img[rgbidx + 2];
        }
    }
}

float cubicInterpolate(float p[4], float x) 
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    if ((output <= 255.0) && (output >= 0.0))
    {
        return output;
    }
    else if (output > 255.0)
    {
        return 255;
    }
    return 0.0;
}

//                               y  x
float bicubicInterpolate(float p[4][4], float y, float x) 
{
    float arr[4];
    arr[0] = cubicInterpolate(p[0], x);
    arr[1] = cubicInterpolate(p[1], x);
    arr[2] = cubicInterpolate(p[2], x);
    arr[3] = cubicInterpolate(p[3], x);
    return cubicInterpolate(arr, y);
}

void bicubicInterpolation(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    //             y  x
    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    int input_x = 0;
    int input_y = 0;

    int window_x = 0;
    int window_y = 0;
    
    int output_x = 0;
    int output_y = 0;

    for (window_y = 0; window_y < 4; window_y++)
    {
        for (window_x = 0; window_x < 4; window_x++)
        {
            window_r[window_y][window_x] = 0;
            window_g[window_y][window_x] = 0;
            window_b[window_y][window_x] = 0;
        }
    }

    for(output_y = 0; output_y < big_height; output_y++)
    {
        for(output_x = 0; output_x < big_width; output_x++)
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

            float r = bicubicInterpolate(window_r, dy, dx);
            float g = bicubicInterpolate(window_g, dy, dx);
            float b = bicubicInterpolate(window_b, dy, dx);

            big_img_data[3 * (output_y * big_width + output_x) + 0] = (unsigned char)r;
            big_img_data[3 * (output_y * big_width + output_x) + 1] = (unsigned char)g;
            big_img_data[3 * (output_y * big_width + output_x) + 2] = (unsigned char)b;
        
        }
    }
}

void nearestNeighbors(unsigned char* big_img_data_gray, unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    int small_x, small_y;

    unsigned char r, g, b;

    for (int y = 0; y < big_height; y++)
    {
        for (int x = 0; x < big_width; x++)
        {
            small_x = x / scale;
            small_y = y / scale;

            r = img_data[3 * (small_y * width + small_x) + 0];
            g = img_data[3 * (small_y * width + small_x) + 1];
            b = img_data[3 * (small_y * width + small_x) + 2];

            big_img_data[3 * (y * big_width + x) + 0] = r;
            big_img_data[3 * (y * big_width + x) + 1] = g;
            big_img_data[3 * (y * big_width + x) + 2] = b;

            big_img_data_gray[(y * big_width + x)] = (21 * r/100) + (71 * g / 100) + (7 * b / 100);
        }
    }
}

void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{

    float img_1_signed = 0;
    float img_2_signed = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img_1_signed = (float)img_1[y * width + x];
            img_2_signed = (float)img_2[y * width + x];
            diff_map[y * width + x] = (float)abs((img_1_signed - img_2_signed) / 255.0); //Normalize 
        }
    }
}

float calculateSSIM(float window1[8][8], float window2[8][8], int window_width, int window_height) {
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    int size = window_height * window_width;
    int valid_count = 0;

    for (int i = 0; i < window_height; ++i) {
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

void SSIM_Grey(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window
    float window_img1[8][8] = { 0 };
    float window_img2[8][8] = { 0 };

    int x_blocks = (width - 1) / 8 + 1;
    int y_blocks = (height - 1) / 8 + 1;

    float ssim_num = 0;

    //For now, generate a smaller image.
    for (int y_blk = 0; y_blk < y_blocks; y_blk++)
    {
        for (int x_blk = 0; x_blk < x_blocks; x_blk++)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))
                    {
                        window_img1[i][j] = img_1[(y_blk * 8 + i) * width + (x_blk * 8 + j)];
                        window_img2[i][j] = img_2[(y_blk * 8 + i) * width + (x_blk * 8 + j)];
                    }
                    else
                    {
                        window_img1[i][j] = -1;
                        window_img2[i][j] = -1;
                    }
                }
            }

            ssim_num = calculateSSIM(window_img1, window_img2, 8, 8);

            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))
                    {
                        ssim_map[(y_blk * 8 + i) * width + (x_blk * 8 + j)] = ssim_num;
                    }
                }
            }
        }
    }
}

void MapMul(float* product_map, float* map_1, float* map_2, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
            product_map[idx] = map_1[idx] * map_2[idx];
        }
    }
}
void Map2Greyscale(unsigned char* grey_img, float* map, int width, int height, int scale)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            grey_img[y * width + x] = (unsigned char)(scale * map[y * width + x]);
        }
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

void GAUSSianBlur_Map(float* blur_map, float* input_map, int width, int height, int radius, float sigma)
{
    //Generate Normalized GAUSSian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float* gaussian_kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size);

    float my_PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062;

    for (int y = 0; y < kernel_size; y++)
    {
        for (int x = 0; x < kernel_size; x++)
        {
            double exponent = -((x - kernel_center) * (x - kernel_center) - (y - kernel_center) * (y - kernel_center)) / (2 * sigma * sigma);
            gaussian_kernel[y * kernel_size + x] = exp(exponent) / (2 * my_PI * sigma * sigma);
            sum += gaussian_kernel[y * kernel_size + x];
        }
    }
    //Normalize
    //May not want to do this as edge cases will not utilize entire kernel.
    //Will try for now. It may be the right way to do it. I don't know for sure.
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            gaussian_kernel[i * kernel_size + j] /= sum;
        }
    }

    //Run through image with kernel centered at current pixel
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            sum = 0.0;
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    int map_y = y + i - radius; //
                    int map_x = x + j - radius;

                    //If we are within the image
                    if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                        sum += input_map[map_y * width + map_x] * gaussian_kernel[i * kernel_size + j];
                    }
                }
            }
            blur_map[y * width + x] = sum;
        }
    }

    free(gaussian_kernel);

}

void MapThreshold(float* map, float threshold, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
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
}
