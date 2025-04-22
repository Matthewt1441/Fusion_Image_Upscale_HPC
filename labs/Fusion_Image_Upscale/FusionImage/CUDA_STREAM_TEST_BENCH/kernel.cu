#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>
#include "util.cu"

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


//Initial Naive approach
__global__ void rgbToRGBA_Kernel(RGBA_t* d_RGBA_img, unsigned char* d_rgb_img, int numpixels)
{
    // Each thread processes one pixel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int sharedIdx = tid * 3;

    // Shared memory for the current block of RGB values
    extern __shared__ unsigned char sharedRGB_char[];

    // Load data into shared memory
    if(idx < numpixels)
    {
        sharedRGB_char[sharedIdx + 0] = d_rgb_img[idx * 3 + 0];
        sharedRGB_char[sharedIdx + 1] = d_rgb_img[idx * 3 + 1];
        sharedRGB_char[sharedIdx + 2] = d_rgb_img[idx * 3 + 2];
    }

    // Synchronize to ensure all threads have loaded their data into shared memory
    __syncthreads();

    // Now process the RGB to RGBA conversion in shared memory
    if(idx < numpixels) {
        // Read RGB values from shared memory
        unsigned char r = sharedRGB_char[sharedIdx + 0];
        unsigned char g = sharedRGB_char[sharedIdx + 1];
        unsigned char b = sharedRGB_char[sharedIdx + 2];

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
    extern __shared__ RGBA_t sharedRGB[];  // Assuming block size is 256 threads

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

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float y, float x)
{
    float arr[4];
    float temp;
    float dx_half   = 0.5 * x;

    temp = p[0][1] + dx_half * (p[0][2] - p[0][0] + x * (2.0 * p[0][0] - 5.0 * p[0][1] + 4.0 * p[0][2] - p[0][3] + x * (3.0 * (p[0][1] - p[0][2]) + p[0][3] - p[0][0])));
    arr[0] = temp;

    temp = p[1][1] + dx_half* (p[1][2] - p[1][0] + x * (2.0 * p[1][0] - 5.0 * p[1][1] + 4.0 * p[1][2] - p[1][3] + x * (3.0 * (p[1][1] - p[1][2]) + p[1][3] - p[1][0])));
    arr[1] = temp;

    temp = p[2][1] + dx_half * (p[2][2] - p[2][0] + x * (2.0 * p[2][0] - 5.0 * p[2][1] + 4.0 * p[2][2] - p[2][3] + x * (3.0 * (p[2][1] - p[2][2]) + p[2][3] - p[2][0])));
    arr[2] = temp;

    temp = p[3][1] + dx_half * (p[3][2] - p[3][0] + x * (2.0 * p[3][0] - 5.0 * p[3][1] + 4.0 * p[3][2] - p[3][3] + x * (3.0 * (p[3][1] - p[3][2]) + p[3][3] - p[3][0])));
    arr[3] = temp;

    temp =  arr[1] + 0.5 * y * (arr[2]  - arr[0]  + y * (2.0 * arr[0]  - 5.0 * arr[1]  + 4.0 * arr[2]  - arr[3]  + y * (3.0 * (arr[1]  - arr[2])  + arr[3]  - arr[0])));

    temp = temp * ((temp < 256.0) && (temp > -1.0)) + 255 * (temp > 255.0);//+ 0 * (temp < 0);
    return temp;
}

//Run with block sizes that are multiples of the scale
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
 
    int g_input_x = 0;
    int g_input_y = 0;
    
    int g_output_x = Col;
    int g_output_y = Row;

    int tile_input_x = 0;
    int tile_input_y = 0;
    
    int window_x = 0;
    int window_y = 0;

    //Only based on Block Size
    int tile_width = (blockDim.x / scale) + 3;
    int tile_height = (blockDim.y / scale) + 3;
    extern __shared__ RGBA_t s_tile[];

    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    RGBA_t rgba_val;

    if(threadIdx.x < tile_width && threadIdx.y < tile_height)
    {
        //Calculate Global Input Index
        g_input_x = blockIdx.x * (blockDim.x / scale) + threadIdx.x - 1;
        g_input_y = blockIdx.y * (blockDim.y / scale) + threadIdx.y - 1;

        // Fill window with Nearest Neighbor edge behavior
        if(g_input_x < 0 || g_input_x >= width)
        {
            // Find nearest in-bounds pixel
            g_input_x = (g_input_x < 0) ? 0 : width - 1;
        }
        // Fill window with Nearest Neighbor edge behavior
        if(g_input_y < 0 || g_input_y >= height)
        {
            // Find nearest in-bounds pixel
            g_input_y = (g_input_y < 0) ? 0 : height - 1;
        }

        s_tile[threadIdx.y * tile_width + threadIdx.x] = img_data[g_input_y * width + g_input_x];
    }
    __syncthreads();

    if(g_output_y < big_height && g_output_x < big_width)
    {
        //Calculate starting index for windows (funky stuff to remove shift)
        float interpolated_x = (((float)threadIdx.x + 0.5f) / (float)scale - 0.5f);
        float interpolated_y = (((float)threadIdx.y + 0.5f) / (float)scale - 0.5f);

        //Round down to nearest index
        int interpolated_idx_x = interpolated_x;
        int interpolated_idx_y = interpolated_y;

        float dx = interpolated_x - interpolated_idx_x;
        float dy = interpolated_y - interpolated_idx_y;

        //Fill local window with tiled input data
        for(window_y = -1; window_y < 3; window_y++)
        {
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image Tile index
                tile_input_x = interpolated_idx_x + window_x + 1;
                tile_input_y = interpolated_idx_y + window_y + 1;

                rgba_val = s_tile[tile_input_y * tile_width + tile_input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        rgba_val.r = (unsigned char)bicubicInterpolateDevice_Shared(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_Shared(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_Shared(window_b, dy, dx);

        big_img_data[g_output_y * big_width + g_output_x] = rgba_val;

        grey_big_img_data[g_output_y * big_width + g_output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

    }

}

__global__ void nearestNeighbors_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int small_x = 0;    int small_y = 0;

    RGBA_t rgba_val;

    if (Row < big_height && Col < big_width)
    {
        small_x = Col / scale;
        small_y = Row / scale;

        rgba_val = img_data[small_y * width + small_x];

        big_img_data[Row * big_width + Col] = rgba_val;

        grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
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

__constant__ float d_gauss_kernel_seperable[7] = { 0.0366328470,   0.111280762,    0.216745317,    0.270682156,    0.216745317,    0.111280762,    0.0366328470 };

__global__ void horizontalGaussianBlurConvolve(float* blur_map, float* input_map, int width, int height, int ksize)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    //Shared Memory based on block size and kernel size (always 7 in our case)
    // Tile Width   = Block_Width + KSize - 1
    // Tile Height  = Block_Height
    extern __shared__ float s_tile_h[];

    int tile_width  = blockDim.x + ksize - 1;
    int tile_height = blockDim.y;
    int radius = ksize/2;


    //Fill Input Tile by striding through the input array
    int tile_input_idx_x = tidx;
    int tile_input_idx_y = tidy;
    while(tile_input_idx_x < tile_width)
    {
        int g_input_idx_x = (tile_input_idx_x - radius) + blockIdx.x * blockDim.x;

        if (g_input_idx_x >= 0 && g_input_idx_x < width) 
        {
            s_tile_h[tile_input_idx_y * tile_width + tile_input_idx_x] = input_map[Row * width + g_input_idx_x];
        }
        else
        {
            s_tile_h[tile_input_idx_y * tile_width + tile_input_idx_x] = 0;
        }

        //Stride by blockDim ammount
        tile_input_idx_x += blockDim.x;
    }
    __syncthreads();


    float sum = 0;
    if (Row < height && Col < width)
    {
        //Define Starting points for input tile
        int tile_y = tidy;
        int tile_x = tidx + radius;

        //Horizontal Convolve
        for (int k = -radius; k <= radius; k++) 
        {
            sum += s_tile_h[tile_y * tile_width + (tile_x - k)] * d_gauss_kernel_seperable[k + radius];
        }

        //Global Write
        blur_map[Row * width + Col] = sum;
    }
}

__global__ void verticalGaussianBlurConvolve(float* blur_map, float* input_map, int width, int height, float threshold, int ksize)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    //Shared Memory based on block size and kernel size (always 7 in our case)
    // Tile Width   = Block_Width
    // Tile Height  = Block_Height + KSize - 1
    extern __shared__ float s_tile_v[];

    int tile_width  = blockDim.x;
    int tile_height = blockDim.y + ksize - 1;
    int radius = ksize/2;

    //Fill Input Tile by striding through the input array
    int tile_input_idx_x = tidx;
    int tile_input_idx_y = tidy;
    while(tile_input_idx_y < tile_height)
    {
        int g_input_idx_y = (tile_input_idx_y - radius) + blockIdx.y * blockDim.y;

        if (g_input_idx_y >= 0 && g_input_idx_y < height) 
        {
            s_tile_v[tile_input_idx_y * tile_width + tile_input_idx_x] = input_map[g_input_idx_y * width + Col];
        }
        else
        {
            s_tile_v[tile_input_idx_y * tile_width + tile_input_idx_x] = 0;
        }

        //Stride by blockDim ammount
        tile_input_idx_y += blockDim.y;
    }
    __syncthreads();


    float sum = 0;
    if (Row < height && Col < width)
    {
        //Define Starting points for input tile
        int tile_y = tidy + radius;
        int tile_x = tidx;

        //Horizontal Convolve
        for (int k = -radius; k <= radius; k++) 
        {
            sum += s_tile_v[(tile_y - k) * tile_width + tile_x] * d_gauss_kernel_seperable[k + radius];
        }

        //Global Write
        blur_map[Row * width + Col] = (sum > threshold) ? 1.0 : 0.0;//sum;
    }
}

__global__ void GaussianBlur_Threshold_Map_Naive_Kernel(float* blur_map, float* input_map, int width, int height, int radius, float sigma, float threshold)
{
    //Generate Normalized Gaussian Kernal for blurring. This may need to be adjusted so I'll make it flexible.
    //We can eventually hardcode this when we settle on ideal blur.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float gaussian_kernel[49] = { 0 };

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
                gaussian_kernel[y * kernel_size + x] = exp(exponent) / (2 * my_PI * sigma * sigma);
                sum += gaussian_kernel[y * kernel_size + x];
            }
        }
        //Normalize
        //May not want to do this as edge cases will not utilize entire kernel.
        //Will try for now. It may be the right way to do it. I don't know for sure.
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                gaussian_kernel[i * kernel_size + j] /= sum;

        sum = 0.0;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int map_y = Row + i - radius; //
                int map_x = Col + j - radius;

                //If we are within the image
                if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                    sum += input_map[map_y * width + map_x] * gaussian_kernel[i * kernel_size + j];
                }
            }
        }

        blur_map[Row * width + Col] = (sum > threshold) ? 1.0 : 0.0;
    }
}

__global__ void Image_Fusion_Kernel_RGBA(RGBA_t* fused_img, RGBA_t* img_1, RGBA_t* img_2, float* weight_map, int width, int height)
{
    //int Row = blockIdx.y * blockDim.y + threadIdx.y;
    //int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //int map_idx = Row * width + Col;
    //int img_idx = 3 * map_idx;
    RGBA_t rgba_pxl1;
    RGBA_t rgba_pxl2;
    RGBA_t rgba_fused;

    //if (Row < height && Col < width)
    if( idx < width*height)
    {
        rgba_pxl1 = img_1[idx];
        rgba_pxl2 = img_2[idx];

        rgba_fused.r = rgba_pxl1.r * weight_map[idx] + rgba_pxl2.r * (1.0 - weight_map[idx]);
        rgba_fused.g = rgba_pxl1.g * weight_map[idx] + rgba_pxl2.g * (1.0 - weight_map[idx]);
        rgba_fused.b = rgba_pxl1.b * weight_map[idx] + rgba_pxl2.b * (1.0 - weight_map[idx]);

        fused_img[idx] = rgba_fused;
    }
}