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

char* readPPM(char* filename, int* width, int* height) {
    //std::ifstream file(filename, std::ios::binary);

    std::ifstream file(filename, std::ios::binary); // open the file and throw exception if it doesn't exist
    if (file.fail())
        throw "File failed to open";

    std::string magicNumber;
    int maxColorValue;
    int w = 0;
    int h = 0;

    file >> magicNumber;
    file >> w >> h >> maxColorValue;

    file.get(); // skip the trailing white space

    size_t size = w * h * 3;
    char* pixel_data = new char[size];

    file.read(pixel_data, size);

    *width = w;
    *height = h;

    return pixel_data;
}

char* readPPMGray(char* filename, int* width, int* height) {
    //std::ifstream file(filename, std::ios::binary);

    std::ifstream file(filename, std::ios::binary); // open the file and throw exception if it doesn't exist
    if (file.fail())
        throw "File failed to open";

    std::string magicNumber;
    int maxColorValue;
    int w = 0;
    int h = 0;

    file >> magicNumber;
    file >> w >> h >> maxColorValue;

    file.get(); // skip the trailing white space

    size_t size = w * h;
    char* pixel_data = new char[size];

    file.read(pixel_data, size);

    *width = w;
    *height = h;

    return pixel_data;
}

void writePPM(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P6" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height) * 3;

    file.write(img_data, size);
}

void writePPMGrey(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P5" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height);

    file.write(img_data, size);
}

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

            if(img1[idx] < img2[idx] - 5 || img1[idx] > img2[idx] + 5 )
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

__device__ float cubicInterpolateDevice_GreyCon(float p[4], float x)
{
    float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

    output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
    return output;
}

__device__ float bicubicInterpolateDevice_GreyCon(float p[4][4], float y, float x)
{
    float arr[4];
    arr[0] = cubicInterpolateDevice_GreyCon(p[0], x);
    arr[1] = cubicInterpolateDevice_GreyCon(p[1], x);
    arr[2] = cubicInterpolateDevice_GreyCon(p[2], x);
    arr[3] = cubicInterpolateDevice_GreyCon(p[3], x);
    return cubicInterpolateDevice_GreyCon(arr, y);
}

__global__ void bicubicInterpolation_GreyCon_Kernel_RGBA(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
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

    RGBA_t rgba_val;

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
                rgba_val = img_data[input_y * width + input_x];

                window_r[window_y + 1][window_x + 1] = (float)rgba_val.r;    //R
                window_g[window_y + 1][window_x + 1] = (float)rgba_val.g;    //G
                window_b[window_y + 1][window_x + 1] = (float)rgba_val.b;    //B
            }
        }

        rgba_val.r = (unsigned char)bicubicInterpolateDevice_GreyCon(window_r, dy, dx);
        rgba_val.g = (unsigned char)bicubicInterpolateDevice_GreyCon(window_g, dy, dx);
        rgba_val.b = (unsigned char)bicubicInterpolateDevice_GreyCon(window_b, dy, dx);

        big_img_data[output_y * big_width + output_x] = rgba_val;

        grey_big_img_data[output_y * big_width + output_x] = 0.21f * rgba_val.r + 0.71f * rgba_val.g + 0.07f * rgba_val.b;

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


// __device__ float cubicInterpolateDevice_Shared(float p[4], float x)
// {
//     float output = p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));

//     output = output * ((output <= 255.0) && (output >= 0.0)) + 255 * (output > 255.0) + 0 * (output < 0);
//     return output;
// }

__device__ float bicubicInterpolateDevice_Shared(float p[4][4], float y, float x)
{
    float arr[4];
    float temp;
    float dx_half   = 0.5 * x;
    float dx_double = 2.0 * x;
    
    temp = p[0][1] + dx_half * (p[0][2] - p[0][0] + dx_double * (p[0][0] - 2.5 * p[0][1] + 4.0 * p[0][2] - p[0][3] + dx_half * (3.0 * (p[0][1] - p[0][2]) + p[0][3] - p[0][0])));
    arr[0] = temp;

    temp = p[1][1] + dx_half * (p[1][2] - p[1][0] + dx_double * (p[1][0] - 2.5 * p[1][1] + 4.0 * p[1][2] - p[1][3] + dx_half * (3.0 * (p[1][1] - p[1][2]) + p[1][3] - p[1][0])));
    arr[1] = temp;

    temp = p[2][1] + dx_half * (p[2][2] - p[2][0] + dx_double * (p[2][0] - 2.5 * p[2][1] + 4.0 * p[2][2] - p[2][3] + dx_half * (3.0 * (p[2][1] - p[2][2]) + p[2][3] - p[2][0])));
    arr[2] = temp;

    temp = p[3][1] + dx_half * (p[3][2] - p[3][0] + dx_double * (p[3][0] - 2.5 * p[3][1] + 4.0 * p[3][2] - p[3][3] + dx_half * (3.0 * (p[3][1] - p[3][2]) + p[3][3] - p[3][0])));
    arr[3] = temp;

    temp =  arr[1] + 0.5 * y * (arr[2]  - arr[0]  + y * (2.0 * arr[0]  - 5.0 * arr[1]  + 4.0 * arr[2]  - arr[3]  + y * (3.0 * (arr[1]  - arr[2])  + arr[3]  - arr[0])));

    temp = temp * ((temp < 256.0) && (temp > -1.0)) + 255 * (temp > 255.0);// + 0 * (temp < 0);
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
        //Calculate starting index for input tile
        float interpolated_x = (float)((threadIdx.x / (scale * 1.0)));
        float interpolated_y = (float)((threadIdx.y / (scale * 1.0)));

        int interpolated_idx_x = (threadIdx.x / scale);
        int interpolated_idx_y = (threadIdx.y / scale);

        float dx = interpolated_x - interpolated_idx_x;
        float dy = interpolated_y - interpolated_idx_y;

        interpolated_idx_x++;   //(threadIdx.x / scale) + 1
        interpolated_idx_y++;   //(threadIdx.y / scale) + 1
    
        //Fill local window with tiled input data
        #pragma unroll
        for(window_y = -1; window_y < 3; window_y++)
        {
            #pragma unroll
            for(window_x = -1; window_x < 3; window_x++)
            {
                //Calculate Input Image Tile index
                tile_input_x = interpolated_idx_x + window_x;
                tile_input_y = interpolated_idx_y + window_y;

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
