#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>
#include "util.cu"


// RGB2Greyscale Function
//
// Parameters:
// pointer to grayscale image
// pointer to rgb image
// width of input image
// height of input image
//
// The function converts the rgb image to grayscale
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

// cubicInterpolate Function
//
// Parameters:
// 4 Pixel array
// dx
//
// The function performs a cubic interpolation along one of the axis.
// Function includes clamping to ensure the pixel is not < 0 or > than 255.
float cubicInterpolate(float p[4], float x) 
{
    //Calculate the interpolated pixel
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    //Clamping: prevent the output pixel value from exceeding 255 or going below 0.
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

// bicubicInterpolate Function
//
// Parameters:
// 4x4 Pixel array
// dy
// dx
//
// The function performs 4 cubic interpolations along one of the axis.
// Those values are then used to interpolate along the other axis
// The funciton calculates the output pixel
float bicubicInterpolate(float p[4][4], float y, float x) 
{
    float arr[4];
    arr[0] = cubicInterpolate(p[0], x);
    arr[1] = cubicInterpolate(p[1], x);
    arr[2] = cubicInterpolate(p[2], x);
    arr[3] = cubicInterpolate(p[3], x);
    return cubicInterpolate(arr, y);
}

// bicubicInterpolation Function
//
// Parameters:
// pointer to larger output image
// width of larger output image
// height of larger output image
// pointer to smaller input image
// width of smaller input image
// height of smaller input image
// scale factor to upscale image by
//
// This function uses the smaller image data to interpolate a larger output image
// It utilizes the bicubic interpolation technique by using a 4x4 window of pixel values to interpolate a number of intermediate points, based on the scale
void bicubicInterpolation(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    // Define Windows for RGB values
    //             y  x
    float window_r[4][4];
    float window_g[4][4];
    float window_b[4][4];

    // Input image x and y coordinates
    int input_x = 0;
    int input_y = 0;

    // Window x and y coordinates
    int window_x = 0;
    int window_y = 0;
    
    // Output x and y coordinates
    int output_x = 0;
    int output_y = 0;

    //Set each window to 0.
    for (window_y = 0; window_y < 4; window_y++)
    {
        for (window_x = 0; window_x < 4; window_x++)
        {
            window_r[window_y][window_x] = 0;
            window_g[window_y][window_x] = 0;
            window_b[window_y][window_x] = 0;
        }
    }

    //For Each Output Pixel
    for(output_y = 0; output_y < big_height; output_y++)
    {
        for(output_x = 0; output_x < big_width; output_x++)
        {
            //Calculate interpolation point
            float interpolated_x = (float)(output_x / (scale * 1.0));
            float interpolated_y = (float)(output_y / (scale * 1.0));

            //Calculate the nearest input pixel index
            int input_block_start_idx_x = (output_x / scale);
            int input_block_start_idx_y = (output_y / scale);

            //Determine how far between input pixels to interpolate
            float dx = interpolated_x - input_block_start_idx_x;
            float dy = interpolated_y - input_block_start_idx_y;

            //Fill 4x4 windows with nearest pixels
            // * = starting input pixel
            // [ ][ ][ ][ ]
            // [ ][*][ ][ ]
            // [ ][ ][ ][ ]
            // [ ][ ][ ][ ]
            for(window_y = -1; window_y < 3; window_y++)
            {
                for(window_x = -1; window_x < 3; window_x++)
                {
                    //Calculate Input Image index
                    input_x = input_block_start_idx_x + window_x;
                    input_y = input_block_start_idx_y + window_y;

                    // Fill window with Nearest Neighbor if input index is out of range
                    if(input_x < 0 || input_x >= width)
                    {
                        // Find nearest in-bounds pixel
                        input_x = (input_x < 0) ? 0 : width - 1;
                    }
                    // Fill window with Nearest Neighbor if input index is out of range
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

            //Use windows to interpolate output pixel
            float r = bicubicInterpolate(window_r, dy, dx);
            float g = bicubicInterpolate(window_g, dy, dx);
            float b = bicubicInterpolate(window_b, dy, dx);

            //Write to output image
            big_img_data[3 * (output_y * big_width + output_x) + 0] = (unsigned char)r;
            big_img_data[3 * (output_y * big_width + output_x) + 1] = (unsigned char)g;
            big_img_data[3 * (output_y * big_width + output_x) + 2] = (unsigned char)b;
        
        }
    }
}

// Nearest Neighbors
//
// Parameters:
// Unsigned Char pointer to upscaled image data
// Unsigned Char pointer to input image data
// Upscaled width and height
// Input width and height
// Integer scale value
//
// The function performs the nearest neighbors algorithm on the input image.

void nearestNeighbors(unsigned char* big_img_data, int big_width, int big_height, unsigned char* img_data, int width, int height, int scale)
{
    int small_x, small_y;                                           // Variables for the x,y coordinates in the original image

    unsigned char r, g, b;                                          // R, G, B Pixel Values

    for (int y = 0; y < big_height; y++)                            // For every x and y pixel values
    {
        for (int x = 0; x < big_width; x++)
        {
            small_x = x / scale;                                    // Calculate the nearest x value
            small_y = y / scale;                                    // Calculaye the nearest y value

            r = img_data[3 * (small_y * width + small_x) + 0];      // Read the R, G, B values
            g = img_data[3 * (small_y * width + small_x) + 1];
            b = img_data[3 * (small_y * width + small_x) + 2];

            big_img_data[3 * (y * big_width + x) + 0] = r;          //Write the R, G, B values
            big_img_data[3 * (y * big_width + x) + 1] = g;
            big_img_data[3 * (y * big_width + x) + 2] = b;
        }
    }
}

// ABS_Difference_Grey
//
// Parameters:
// Float pointer to the difference map
// Unsigned Char pointer to upscaled image 1
// Unsigned Char pointer to upscaled image 2
// Upscaled width and height
//
// The function subtracts the two images and takes the absolute value of the difference.

void ABS_Difference_Grey(float* diff_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{

    float img_1_signed = 0;                                                                 // img1 variable
    float img_2_signed = 0;                                                                 // img2 variable

    for (int y = 0; y < height; y++)                                                        // for every pixel
    {
        for (int x = 0; x < width; x++)
        {
            img_1_signed = (float)img_1[y * width + x];                                     // case img1 to float        
            img_2_signed = (float)img_2[y * width + x];                                     // case img2 to float
            diff_map[y * width + x] = (float)abs((img_1_signed - img_2_signed) / 255.0);    // subtract the pixel values
        }                                                                                   // and normalize by the max value
    }
}

// calculateSSIM
//
// Parameters:
// Float pointer to the image 1 window
// Float pointer to the image 2 window
// Integer of the window height and width
//
// calculateSSIM takes the two image windows and calculates the SSIM index

float calculateSSIM(float window1[8][8], float window2[8][8], int window_width, int window_height) 
{
    //Running variables to calculate the sum of the images, and sqaure values
    float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, sum12 = 0;

    //Total pixel count
    int size = window_height * window_width;

    //Number of valid pixels in the window
    int valid_count = 0;

    for (int i = 0; i < window_height; ++i)                     //For every pixel in the window
    {
        for (int j = 0; j < window_width; ++j)
        {
            if ((window1[i][j] >= 0) && (window2[i][j] >= 0))   //If the pixel is valid (non negative)
            {
                sum1 += window1[i][j];                          //add img 1 value
                sum2 += window2[i][j];                          //add img 2 value
                sum1Sq += window1[i][j] * window1[i][j];        //add img 1 ^2
                sum2Sq += window2[i][j] * window2[i][j];        //add img 2 ^2
                sum12 += window1[i][j] * window2[i][j];         //add img 1 * img 2
                valid_count++;                                  //increment valid count
            }
        }
    }

    //Calculate mean, sigma, etc for SSIM calculation
    float mu1 = sum1 / valid_count;
    float mu2 = sum2 / valid_count;
    float sigma1Sq = (sum1Sq / valid_count) - (mu1 * mu1);
    float sigma2Sq = (sum2Sq / valid_count) - (mu2 * mu2);
    float sigma12 = (sum12 / valid_count) - (mu1 * mu2);

    // Stabilizing constants
    float C1 = 6.5025;  // (K1*L)^2, where K1=0.01 and L=255
    float C2 = 58.5225; // (K2*L)^2, where K2=0.03 and L=255

    //Compute the SSIM
    float ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2));
    return ssim;
}

// SSIM_Grey
//
// Parameters:
// Float pointer to the SSIM map
// Unsigned Char pointer to upscaled image 1
// Unsigned Char pointer to upscaled image 2
// Upscaled width and height
//
// The function calculates the SSIM map using 8x8 windows.

void SSIM_Grey(float* ssim_map, unsigned char* img_1, unsigned char* img_2, int width, int height)
{
    float window_img1[8][8] = { 0 };                    // Window array for img 1
    float window_img2[8][8] = { 0 };                    // Window array for img 2 

    int x_blocks = (width - 1) / 8 + 1;                 // Number of SSIM windows for the x axis
    int y_blocks = (height - 1) / 8 + 1;                // Number of SSIM windows for the y axis

    float ssim_num = 0;                                 // SSIM Value

    for (int y_blk = 0; y_blk < y_blocks; y_blk++)      //For every block in the x and y axis  
    {
        for (int x_blk = 0; x_blk < x_blocks; x_blk++)
        {
            for (int i = 0; i < 8; i++)                                                         //For every pixel in the window
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))                //If the pixel is within the image
                    {
                        window_img1[i][j] = img_1[(y_blk * 8 + i) * width + (x_blk * 8 + j)];   //Store img 1 pixels
                        window_img2[i][j] = img_2[(y_blk * 8 + i) * width + (x_blk * 8 + j)];   //Store img 2 pixels
                    }
                    else                                                                        //If the pixel is out of bounds
                    {
                        window_img1[i][j] = -1;                                                 //Store pixel value as -1 (invalid)
                        window_img2[i][j] = -1;
                    }
                }
            }

            ssim_num = calculateSSIM(window_img1, window_img2, 8, 8);                           //Calculate the SSIM number

            for (int i = 0; i < 8; i++)                                                         //For every pixel in the window
            {
                for (int j = 0; j < 8; j++)
                {
                    if ((x_blk * 8 + j) < (width) && ((y_blk * 8 + i) * height))                //If the pixel is out of bounds
                    {
                        ssim_map[(y_blk * 8 + i) * width + (x_blk * 8 + j)] = ssim_num;         //Store the SSIM value into the map
                    }
                }
            }
        }
    }
}

// MapMul
//
// Parameters:
// Float pointer to the Artifact (Product) map
// Float pointer to difference map
// Float pointer to ssim map
// Upscaled width and height
//
// The function calculates the SSIM map using 8x8 windows.

void MapMul(float* product_map, float* map_1, float* map_2, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)                    //For every pixel in the map
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
            product_map[idx] = map_1[idx] * map_2[idx]; //Multiply the two maps together
        }
    }
}

// Map2Greyscale
//
// Parameters:
// Float pointer to the Artifact (Product) map
// Float pointer to map
// Float pointer to ssim map
// Upscaled width and height
// Scale (Max Pixel Value)
//
// The function takes every value in the map and converts it to a pixel value between 0-255.

void Map2Greyscale(unsigned char* grey_img, float* map, int width, int height, int scale)
{
    for (int y = 0; y < height; y++)        //For every pixel
    {
        for (int x = 0; x < width; x++)
        {
            grey_img[y * width + x] = (unsigned char)(scale * map[y * width + x]);  // Multiply the map by the scale value and case to a char
        }
    }
}

// GaussianBlur_Map
//
// Parameters:
// Float pointer to the output blurred map
// Float pointer to the input map
// Upscaled width and height
// radius and sigma used for guassian blur kernel
void GaussianBlur_Map(float* blur_map, float* input_map, int width, int height, int radius, float sigma)
{
    //Generate Normalized Gaussian Kernal for blurring.
    int kernel_size = 2 * radius + 1;
    int kernel_center = kernel_size / 2;
    float sum = 0.0;
    float* gaussian_kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size);
    float my_PI = 3.1415926;
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
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            gaussian_kernel[i * kernel_size + j] /= sum;
        }
    }


    //Convolve image with kernel centered at current pixel
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

// MapThreshold
//
// Parameters:
// Float pointer to the Artifact map
// Float for the threshold
// Upscaled width and height
//
// The MapThreshold binarizes the artifact map based on a threshold.

void MapThreshold(float* map, float threshold, int width, int height)
{
    int idx = 0;
    for (int y = 0; y < height; y++)        //For every pixel
    {
        for (int x = 0; x < width; x++)
        {
            idx = y * width + x;
            if (map[idx] > threshold)       //If the value is above the threshold
            {
                map[idx] = 1.0;             //Set to 1
            }
            else
            {
                map[idx] = 0.0;             //Otherwise, set to 0
            }
        }
    }
}

// Image_Fusion
//
// Parameters:
// Unsigned Char Fused Image
// Unsigned Char Upscaling Image Method 1
// Unsigned Char Upscaling Image Method 2
// Float pointer Artifact Map
// Upscaled width and height
//
// The fusion algorithm takes the two upscaled images and multiplies one image by the weight map and the other by its inverse

void Image_Fusion(unsigned char* fused_img, unsigned char* img_1, unsigned char* img_2, float* weight_map, int width, int height)
{
    for (int y = 0; y < height; y++)        //For every pixel
    {
        for (int x = 0; x < width; x++)
        {
            int map_idx = (y * width + x);
            int img_idx = 3 * map_idx;

            //For each pixel color
            //Set each fused image pixel to one of the images upscaling methods.
            //The first image is multiplied by the weight map and the other by the inverse (1-Weight_Map).
            
            fused_img[img_idx + 0] = img_1[img_idx + 0] * weight_map[map_idx] + img_2[img_idx + 0] * (1.0 - weight_map[map_idx]);
            fused_img[img_idx + 1] = img_1[img_idx + 1] * weight_map[map_idx] + img_2[img_idx + 1] * (1.0 - weight_map[map_idx]);
            fused_img[img_idx + 2] = img_1[img_idx + 2] * weight_map[map_idx] + img_2[img_idx + 2] * (1.0 - weight_map[map_idx]);
        }
    }
}
     