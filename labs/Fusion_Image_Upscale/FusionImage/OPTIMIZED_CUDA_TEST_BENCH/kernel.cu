#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>
#include "util.cu"

// RGB2Greyscale Kernel
//
// Parameters:
// pointer to rgb image
// pointer to grayscale image
// width of input image
// height of input image
//
// The function converts the rgb image to grayscale
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


// rgbToRGBA_Kernel
//
// Parameters:
// pointer to array of RGBA structs
// pointer to rgb image
// total number of pixels
//
// The function converts the rgb array to an array of RGBA structs. These structs are padded to be 32-bits in length.
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


// rgbaToRGB_Kernel
//
// Parameters:
// pointer to rgb image
// pointer to array of RGBA structs
// total number of pixels
//
// The function converts the rgba struct array to an array of RGB values.
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

// BiCubic Interpolate Device Function
//
// Parameters:
// 4x4 Pixel array
// dy, dx
//
// The device function calculates the interpolated images based off the 4x4 array of pixels
// First it interpolates across the points in the x axis to generate 4 points.
// These new points are then used to inteprolate across the y axis to generate the final pixel.
// Function includes clamping to ensure the pixel is not < 0 or > than 255.

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float y, float x)
{
    float arr[4];               //x interpolations array
    float temp;
    float dx_half   = 0.5 * x;  //repeated calculation, saved to local register.

    //interpolate across the x axis pixels.
    temp = p[0][1] + dx_half * (p[0][2] - p[0][0] + x * (2.0 * p[0][0] - 5.0 * p[0][1] + 4.0 * p[0][2] - p[0][3] + x * (3.0 * (p[0][1] - p[0][2]) + p[0][3] - p[0][0])));
    arr[0] = temp;

    temp = p[1][1] + dx_half* (p[1][2] - p[1][0] + x * (2.0 * p[1][0] - 5.0 * p[1][1] + 4.0 * p[1][2] - p[1][3] + x * (3.0 * (p[1][1] - p[1][2]) + p[1][3] - p[1][0])));
    arr[1] = temp;

    temp = p[2][1] + dx_half * (p[2][2] - p[2][0] + x * (2.0 * p[2][0] - 5.0 * p[2][1] + 4.0 * p[2][2] - p[2][3] + x * (3.0 * (p[2][1] - p[2][2]) + p[2][3] - p[2][0])));
    arr[2] = temp;

    temp = p[3][1] + dx_half * (p[3][2] - p[3][0] + x * (2.0 * p[3][0] - 5.0 * p[3][1] + 4.0 * p[3][2] - p[3][3] + x * (3.0 * (p[3][1] - p[3][2]) + p[3][3] - p[3][0])));
    arr[3] = temp;

    //interpolate across the y axis with the points generated from the x interpolations.
    temp =  arr[1] + 0.5 * y * (arr[2]  - arr[0]  + y * (2.0 * arr[0]  - 5.0 * arr[1]  + 4.0 * arr[2]  - arr[3]  + y * (3.0 * (arr[1]  - arr[2])  + arr[3]  - arr[0])));

    //Clamping for the pixel value. Rather than using if-else statements, we multiply the float value by the condition.
    //Avoids branching but adds additional computation.
    temp = temp * ((temp < 256.0) && (temp > -1.0)) + 255 * (temp > 255.0);//+ 0 * (temp < 0);
    return temp;
}

// bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA
//
// Parameters:
// pointer to larger output image
// pointer to larger output gray image
// pointer to smaller input image
// width of larger output image
// height of larger output image
// width of smaller input image
// height of smaller input image
// scale factor to upscale image by
//
// This kernel uses the smaller image data to interpolate a larger output image
// It utilizes the bicubic interpolation technique by using a 4x4 window of pixel values to interpolate a number of intermediate points, based on the scale
// This kernel utilizes shared memory and is expected to be ran with block sizes that are multiples of the scale.
// Shared Memory Size if based on the block size
__global__ void bicubicInterpolation_Shared_Memory_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    
    //Calculate the output pixel row and col for the thread
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
 
    // Input image x and y coordinates
    int g_input_x = 0;
    int g_input_y = 0;
    
    // Output image x and y coordinates
    int g_output_x = Col;
    int g_output_y = Row;

    // X and Y coordinates of the shared memory tile
    int tile_input_x = 0;
    int tile_input_y = 0;
    
    // X and Y coordinates for the window used for interpolation
    int window_x = 0;
    int window_y = 0;

    //Calculate the width and heigh of the shared memory tile
    //Becuae kernel is ran with a scale multiple block dimension, this is only based on Block Size
    int tile_width = (blockDim.x / scale) + 3;
    int tile_height = (blockDim.y / scale) + 3;
    extern __shared__ RGBA_t s_tile[];

    //4x4 windows each thread will use to interpolate over
    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    //Temp RGBA val
    RGBA_t rgba_val;

    //Fill Shared memory with input pixels
    if(threadIdx.x < tile_width && threadIdx.y < tile_height)
    {
        //Calculate Global Input Index
        g_input_x = blockIdx.x * (blockDim.x / scale) + threadIdx.x - 1;
        g_input_y = blockIdx.y * (blockDim.y / scale) + threadIdx.y - 1;

        // Fill tile with Nearest Neighbor if input index is out of range
        if(g_input_x < 0 || g_input_x >= width)
        {
            // Find nearest in-bounds pixel
            g_input_x = (g_input_x < 0) ? 0 : width - 1;
        }
        // Fill tile with Nearest Neighbor if input index is out of range
        if(g_input_y < 0 || g_input_y >= height)
        {
            // Find nearest in-bounds pixel
            g_input_y = (g_input_y < 0) ? 0 : height - 1;
        }

        s_tile[threadIdx.y * tile_width + threadIdx.x] = img_data[g_input_y * width + g_input_x];
    }
    __syncthreads();

    //For each output pixel within range
    if(g_output_y < big_height && g_output_x < big_width)
    {
        //Calculate interpolation point
        //Add shift to prevent final upscaled image from being shifted
        float interpolated_x = (((float)threadIdx.x + 0.5f) / (float)scale - 0.5f);
        float interpolated_y = (((float)threadIdx.y + 0.5f) / (float)scale - 0.5f);

        //Calculate the nearest input pixel index
        int interpolated_idx_x = interpolated_x;
        int interpolated_idx_y = interpolated_y;

        //Determine how far between input pixels to interpolate
        float dx = interpolated_x - interpolated_idx_x;
        float dy = interpolated_y - interpolated_idx_y;

        //Fill local window with tiled input data, this will always be inbound as tile is padded
        for(window_y = -1; window_y < 3; window_y++)
        {
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image Tile index
                tile_input_x = interpolated_idx_x + window_x + 1;
                tile_input_y = interpolated_idx_y + window_y + 1;

                //Read pixel value
                rgba_val = s_tile[tile_input_y * tile_width + tile_input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        //Use windows to interpolate output pixel
        rgba_val.r = (unsigned char)bicubicInterpolateDevice_Shared(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_Shared(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_Shared(window_b, dy, dx);

        //Write to output image
        big_img_data[g_output_y * big_width + g_output_x] = rgba_val;

        //Convert RGBA value to gray scale
        grey_big_img_data[g_output_y * big_width + g_output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

    }

}

// Nearest Neighbors + Greyscale Conversion With RGBA Struct
//
// Parameters:
// RGBA pointer to upscaled image data
// Unsigned Char pointer to greyscaled upscaled image data
// RGBA pointer to input image data
// Upscaled width and height
// Input width and height
// Integer scale value
//
// The kernel performs the nearest neighbors algorithm on the input image. Calculates the greyscale pixel value after upscaling.
// Utilizes the RGBA struct to improve data coalescing

__global__ void nearestNeighbors_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    //Calculate the output pixel row and col for the thread
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //Variables to store the x, y coordinates of the input image
    int small_x = 0;    int small_y = 0;

    //Variable to store the reference pixel value
    RGBA_t rgba_val;

    //If the thread is within the range of the output image
    if (Row < big_height && Col < big_width)
    {
        //Calculate the x,y coordinates for the input image (nearest neighbors)
        small_x = Col / scale;
        small_y = Row / scale;

        //Read the data from the input image
        rgba_val = img_data[small_y * width + small_x];

        //Store data into the output array
        big_img_data[Row * big_width + Col] = rgba_val;

        //Calculate the greyscale image
        grey_big_img_data[Row * big_width + Col] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;
    }
}

// Artifact Detection (SSIM & Difference Map) Using Shared Memory
//
// Parameters:
// Float pointer to artifact map
// Unsigned Char pointer to first greyscaled upscaled image data
// Unsigned Char pointer to second greyscaled upscaled image data
// Image width and height
//
// The kernel performs the artifact map generation by calcualting the difference between the two upscaled images and then
// calculating the SSIM for an 8x8 window.
// All threads help store pixel data into shared memory so all threads can work together to compute the SSIM.

// Parameters to SSIM calculation. Made as defines to reduce the need to compute it multiple times.
// SSIM calculation uses 8x8 windows which transfers to the block dimension.
#define WINDOW_SIZE     8
#define WINDOW_PIXELS   64

__global__ void Artifact_Shared_Memory_Kernel(float* artifact_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{

    //Shared Memory for 8x8 image data. Stores the image data for both upscaled images.
    // [         [IMG 1]                      [IMG 2]       ]
    // INDEX 127.........INDEX 64....INDEX 63.........INDEX 0

    extern __shared__ float window_img[];

    //Calculate artifact map output coordinates
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //Store the TID for future use.
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    //Intermediate registers for calculating SSIM
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;
    float img_diff;

    //Stores the amount of valid pixels when calcualting the mean of the 8x8 window.
    int valid_count = 0;

    //If the thread is within the output image
    if (Row < height && Col < width)
    {
        //Read pixel values from the two images
        sum1 = img_1[Row * width + Col];    //Note: Using these as temp registers. Just pretend they are called temp1 & temp2
        sum2 = img_2[Row * width + Col];

        //Store pixel values into the shared memory
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = sum1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = sum2;
        
        //Calculate the pixel difference
        img_diff = (float)abs((sum1 - sum2) / 255.0);
    }
    // If the thread is out of bounds, store the pixel value as -1 to signify invalid pixel.
    else
    {
        window_img[tid_y * WINDOW_SIZE + tid_x + WINDOW_PIXELS] = -1;
        window_img[tid_y * WINDOW_SIZE + tid_x] = -1;
    }

    sum1 = 0; sum2 = 0; //reset registers

    __syncthreads();    //Sync point to ensure the 8x8 window in the shared memory is fully populated.

    //8x8 Window For Loop
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            //If the image pixel is valid (>-1)
            if ((window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] >= 0) && (window_img[i * WINDOW_SIZE + j] >= 0))
            {
                sum1 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];                                                        //IMG1 Sum
                sum2 += window_img[i * WINDOW_SIZE + j];                                                                        //IMG2 Sum
                sum1Sq += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS];    //IMG1 ^ 2 Sum    
                sum2Sq += window_img[i * WINDOW_SIZE + j] * window_img[i * WINDOW_SIZE + j];                                    //IMG2 ^ 2 Sum
                sum12 += window_img[i * WINDOW_SIZE + j + WINDOW_PIXELS] * window_img[i * WINDOW_SIZE + j];                     //IMG1 * IMG2 Sum
                valid_count++;                                                                                                  //# of valid pixels
            }
        }
    }

    //Calculate the mean of img1 and img2
    float mu1 = sum1 / valid_count;
    float mu2 = sum2 / valid_count;

    //Calculate sigma for the imgs
    float sigma1Sq = (sum1Sq / valid_count) - (mu1 * mu1);
    float sigma2Sq = (sum2Sq / valid_count) - (mu2 * mu2);
    float sigma12 = (sum12 / valid_count) - (mu1 * mu2);

    // Stabilizing constants
    float C1 = 6.5025;          // (K1*L)^2, where K1=0.01 and L=255
    float C2 = 58.5225;         // (K2*L)^2, where K2=0.03 and L=255

    //SSIM Equation
    float ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2));

    //Calculate the artifact map pixel by multiplying the ssim index with the difference between the two images
    artifact_map[Row * width + Col] = ssim * img_diff;
}

// Define in Constant Memory the 1D Guassian Blur Kernel
__constant__ float d_gauss_kernel_seperable[7] = { 0.0366328470,   0.111280762,    0.216745317,    0.270682156,    0.216745317,    0.111280762,    0.0366328470 };


// horizontalGaussianBlurConvolve
//
// Parameters:
// Float pointer to output blur map
// Float pointer to input artifact map
// Image width and height
// Size of Guassian Blur Kernel (always 7)
//
// The kernel performs the horizontal convolution of the guassian blur across the input artifact map.
// All threads help store input data into shared memory.
// This kernel is launched with 1 thread per output
__global__ void horizontalGaussianBlurConvolve(float* blur_map, float* input_map, int width, int height, int ksize)
{
    //Output Coordinates and Input thread id
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
        // Calculate input x index. 
        int g_input_idx_x = (tile_input_idx_x - radius) + blockIdx.x * blockDim.x;

        // Fill Tile with 0 if input index is out of range
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


// verticalGaussianBlurConvolve
//
// Parameters:
// Float pointer to output blur map
// Float pointer to partially blurred input map
// Image width and height
// Size of Guassian Blur Kernel (always 7)
//
// The kernel performs the vertical convolution of the guassian blur across the input artifact map.
// All threads help store input data into shared memory.
// This kernel is launched with 1 thread per output
__global__ void verticalGaussianBlurConvolve(float* blur_map, float* input_map, int width, int height, float threshold, int ksize)
{
    //Output Coordinates and Input thread id
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
        // Calculate input y index. 
        int g_input_idx_y = (tile_input_idx_y - radius) + blockIdx.y * blockDim.y;

        // Fill Tile with 0 if input index is out of range
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

// Image_Fusion_Kernel_RGBA
//
// Parameters:
// pointer to Fused Image
// pointer to Upscaling Image Method 1
// pointer to Upscaling Image Method 2
// Float pointer Artifact Map
// Upscaled width and height
//
// The fusion algorithm takes the two upscaled images and multiplies one image by the weight map and the other by its inverse
__global__ void Image_Fusion_Kernel_RGBA(RGBA_t* fused_img, RGBA_t* img_1, RGBA_t* img_2, float* weight_map, int width, int height)
{
    //Output index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Pixel parameters
    RGBA_t rgba_pxl1;
    RGBA_t rgba_pxl2;
    RGBA_t rgba_fused;

    if( idx < width*height)
    {
        //Read from both input images
        rgba_pxl1 = img_1[idx];
        rgba_pxl2 = img_2[idx];

        //Use weight map to fuse two images
        rgba_fused.r = rgba_pxl1.r * weight_map[idx] + rgba_pxl2.r * (1.0 - weight_map[idx]);
        rgba_fused.g = rgba_pxl1.g * weight_map[idx] + rgba_pxl2.g * (1.0 - weight_map[idx]);
        rgba_fused.b = rgba_pxl1.b * weight_map[idx] + rgba_pxl2.b * (1.0 - weight_map[idx]);

        //Write output pixel
        fused_img[idx] = rgba_fused;
    }
}