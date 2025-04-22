#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

typedef struct __align__(4) { // Or alignas(4) in C++11
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a; // Padding to ensure 4-byte alignment
}RGBA_t;

__global__ void RGB2GreyscaleKernel(unsigned char* rgb_img, unsigned char* grey_img, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        int rgbidx = rgbidx = 3 * (Row * width + Col);
        grey_img[Row * width + Col] = (21 * rgb_img[rgbidx + 0] /100) + (71 * rgb_img[rgbidx + 1] / 100) + (7 * rgb_img[rgbidx + 2] / 100);
    }
}

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

//Initial Naive approach
__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int sharedIdx = tid * 3;

    // Shared memory for the current block of RGB values
    __shared__ unsigned char sharedRGB[256 * 3];  // Assuming block size is 256 threads

    // Load data into shared memory
    if(idx < numpixels)
    {
        sharedRGB[sharedIdx + 0] = d_rgb_img[idx * 3 + 0];
        sharedRGB[sharedIdx + 1] = d_rgb_img[idx * 3 + 1];
        sharedRGB[sharedIdx + 2] = d_rgb_img[idx * 3 + 2];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGB to RGBA conversion in shared memory
    if(idx < numpixels) {
        // Read RGB values from shared memory
        unsigned char r = sharedRGB[sharedIdx + 0];
        unsigned char g = sharedRGB[sharedIdx + 1];
        unsigned char b = sharedRGB[sharedIdx + 2];

        // Write to RGBA array (global memory)
        d_RGBA_img[idx].r = r;
        d_RGBA_img[idx].g = g;
        d_RGBA_img[idx].b = b;
        d_RGBA_img[idx].a = 255;  // Alpha is fully opaque
    }
}


//Initial Naive approach
__global__ void rgbaToRGB_Kernel(unsigned char* d_rgb_img, RGBA_t* d_rgba_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sharedIdx = threadIdx.x;

    // Shared memory for the current block of RGBA values
    __shared__ RGBA_t sharedRGB[256];  // Assuming block size is 256 threads

    // Load data into shared memory
    if(idx < numpixels)
    {
        sharedRGB[sharedIdx] = d_rgba_img[idx];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGBA to RGB conversion in shared memory
    if(idx < numpixels) {
        // Read RGBA values from shared memory
        RGBA_t rgba_val = sharedRGB[sharedIdx];

        // Write to RGB array (global memory)
        d_rgb_img[idx * 3 + 0] = rgba_val.r;
        d_rgb_img[idx * 3 + 1] = rgba_val.g;
        d_rgb_img[idx * 3 + 2] = rgba_val.b;
    }
}
void Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height)
{
    int idx = 0;
    int y;
    int x;
    bool pass = true;
    for( y = 0; y < height; y++) 
    {
        for(x = 0; x < width; x++)
        {
            idx = y * width + x;
            char img1_r = img1[idx + 0];
            char img1_g = img1[idx + 1];
            char img1_b = img1[idx + 2];
            char img2_r = img2[idx + 0];
            char img2_g = img2[idx + 1];
            char img2_b = img2[idx + 2];

            if((img2_r < img1_r - 5) || (img2_r > img1_r + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

            if((img2_g < img1_g - 5) || (img2_g > img1_g + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

            if((img2_b < img1_b - 5) || (img2_b > img1_b + 5))
            {
                pass = false;
                goto LOOP_EXIT;
            }

            ////                 R                                   G                                   B
            //if((img1[idx + 0] != img2[idx + 0]) || (img1[idx + 1] != img2[idx + 1]) || (img1[idx + 2] != img2[idx + 2]))
            //{
            //    pass = false;
            //    goto LOOP_EXIT;
            //}


        }
    }

LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d, Img1 [%d, %d, %d], Img2 [%d, %d, %d]\n", x, y, img1[idx + 0], img1[idx + 1], img1[idx + 2], img2[idx + 0], img2[idx + 1], img2[idx + 2]);

    }
    else
    {
        printf("Images match!\n");
    }

}

void Grey_Image_Compare(unsigned char* img1, unsigned char* img2, int width, int height)
{
    int idx = 0;
    int y;
    int x;
    bool pass = true;
    for( y = 0; y < height; y++) 
    {
        for(x = 0; x < width; x++)
        {
            idx = y * width + x;

            if(img1[idx] < img2[idx] - 10 || img1[idx] > img2[idx] + 10 )
            {
                pass = false;
                goto GREY_LOOP_EXIT;
            }


        }
    }

GREY_LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d, Img1 [%d, %d, %d], Img2 [%d, %d, %d]\n", x, y, img1[idx + 0], img1[idx + 1], img1[idx + 2], img2[idx + 0], img2[idx + 1], img2[idx + 2]);

    }
    else
    {
        printf("Images match!\n");
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

__global__ void SSIM_Grey_Kernel(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window
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


__device__ float calculateSSIM_Device(float window1[8][8], float window2[8][8], int window_width, int window_height)
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

__global__ void Artifact_Grey_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window
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

        artifact_map[Row * width + Col] = calculateSSIM_Device(window_img1, window_img2, 8, 8) * (float)abs((window_img1[0][0] - window_img2[0][0]) / 255.0);
    }
}

#define WINDOW_SIZE     8
#define WINDOW_PIXELS   64

__global__ void Artifact_Shared_Memory_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window

    extern __shared__ float window_img[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    int valid_count = 0;

    //For now, generate a smaller image.

    if (Row < height && Col < width)
    {
        sum1 = img_1[Row * width + Col];    //Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = sum1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = sum2;
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }

    else
    {
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = -1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if ((window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] >= 0) && (window_img[i * WINDOW_SIZE + j] >= 0))
            {
                sum1 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];
                sum2 += window_img[i * WINDOW_SIZE + j];
                sum1Sq += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];
                sum2Sq += window_img[i * WINDOW_SIZE + j] * window_img[i * WINDOW_SIZE + j];
                sum12 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j];
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

    artifact_map[Row * width + Col] = ssim * img_diff;
}


#define WINDOW_SIZE2     16
#define WINDOW_PIXELS2   256


__global__ void Artifact_Shared_Memory_Kernel3(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window

    extern __shared__ float window_img[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    int valid_count = 0;

    //For now, generate a smaller image.

    if (Row < height && Col < width)
    {
        sum1 = img_1[Row * width + Col];    //Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];
        window_img[tid_y * WINDOW_SIZE2 + tid_x + WINDOW_PIXELS2] = sum1;
        window_img[tid_y * WINDOW_SIZE2 + tid_x] = sum2;
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }

    else
    {
        window_img[tid_y * WINDOW_SIZE2 + tid_x + WINDOW_PIXELS] = -1;
        window_img[tid_y * WINDOW_SIZE2 + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();

    int offset_x = 8 * (tid_x >= 8);
    int offset_y = 8 * (tid_y >= 8);

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if ((window_img[(i + offset_y) * WINDOW_SIZE2 + j + WINDOW_PIXELS2 + offset_x] >= 0) && (window_img[(i + offset_y) * WINDOW_SIZE + j + offset_x] >= 0))
            {
                sum1 += window_img[(i + offset_y) * WINDOW_SIZE2 + j + WINDOW_PIXELS2 + offset_x];
                sum2 += window_img[(i + offset_y) * WINDOW_SIZE2 + j + offset_x];
                sum1Sq += window_img[(i + offset_y) * WINDOW_SIZE2 + j + WINDOW_PIXELS2 + offset_x] * window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS2 + offset_x];
                sum2Sq += window_img[(i + offset_y) * WINDOW_SIZE2 + j + offset_x] * window_img[i * WINDOW_SIZE + j + offset_x];
                sum12 += window_img[(i + offset_y) * WINDOW_SIZE2 + j + WINDOW_PIXELS2 + offset_x] * window_img[i * WINDOW_SIZE + j + offset_x];
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

    artifact_map[Row * width + Col] = ssim * img_diff;
}

__global__ void Artifact_Shared_Memory_Kernel2(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    //int window_size = 8;
    //Window size dictates the size of structures that we can detect. Maybe should look into what effect this has
    //on overall image quality & performance
    // Consider the gaussian option with an 11x11 window

    extern __shared__ float window_img[];

    __shared__ float ssim_sums[5];  //sum1, sum2, sum1Sq, sum2Sq, sum12
    //__shared__ float ssim_offsets[6];
    int ssim_offset_img1[6] = { WINDOW_PIXELS, 0, WINDOW_PIXELS, 0, WINDOW_PIXELS };
    int ssim_offset_img2[6] = { WINDOW_PIXELS, 0, WINDOW_PIXELS, 0, 0 };

    __shared__ float ssim;

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    int valid_count = 0;

    //For now, generate a smaller image.

    if (Row < height && Col < width)
    {
        sum1 = img_1[Row * width + Col];    //Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];
        window_img[tid_y * WINDOW_SIZE + tid_x + (WINDOW_PIXELS)] = sum1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = sum2;
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }

    else
    {
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = -1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();

    if (tid_x < 5 && tid_y == 0)
    {
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                if ((window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] >= 0) && (window_img[i * WINDOW_SIZE + j] >= 0))
                {
                    ssim_sums[tid_x] += (window_img[i * WINDOW_SIZE + j + ssim_offset_img1[tid_x]] * (window_img[i * WINDOW_SIZE + j + ssim_offset_img2[tid_x]]) * (tid_x > 1) + (tid_x < 2));
                    valid_count++;
                }
            }
        }
    }
    __syncthreads();

    if (tid_x == 0 && tid_y == 0)
    {
        float mu1 = ssim_sums[0] / valid_count;
        float mu2 = ssim_sums[1] / valid_count;
        float sigma1Sq = (ssim_sums[2] / valid_count) - (mu1 * mu1);
        float sigma2Sq = (ssim_sums[3] / valid_count) - (mu2 * mu2);
        float sigma12 = (ssim_sums[4] / valid_count) - (mu1 * mu2);

        // Stabilizing constants
        float C1 = 6.5025; // (K1*L)^2, where K1=0.01 and L=255
        float C2 = 58.5225; // (K2*L)^2, where K2=0.03 and L=255

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2));
    }
    __syncthreads();

    artifact_map[Row * width + Col] = ssim * img_diff;
}