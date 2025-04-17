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

        grey_big_img_data[Row * big_width + Col] = (21 * rgba_val.r / 100) + (71 * rgba_val.g /100) + (7  * rgba_val.b /100);
    }
}


////Nearest Neighbors but the shared memory uses one thread to output a pixel.
__global__ void nearestNeighbors_shared_memory_one_thread_per_pixel_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    extern __shared__ RGBA_t img_pixels[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int SHARE_MEM_WIDTH = blockDim.x / scale;

    int big_x = 0;    int big_y = 0;

    RGBA_t rgba_val;

    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
    if (Row < big_height && Col < big_width)
    {
        if (tid_y < SHARE_MEM_WIDTH && tid_x < SHARE_MEM_WIDTH)
        {
            img_pixels[tid_y * (SHARE_MEM_WIDTH) + tid_x] = img_data[((blockIdx.y * blockDim.y)/scale + tid_y) * width + ((blockIdx.x * blockDim.x)/scale) + tid_x];
        }
    }
    //else
    //{
    //    img_pixels[tid_y * blockDim.x + tid_x] = 0;
    //}

    __syncthreads();

    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
    if (Row < big_height && Col < big_width)
    {
        rgba_val = img_pixels[(tid_y / scale) * SHARE_MEM_WIDTH + (tid_x / scale)];

        big_img_data[Row * big_width + Col] = rgba_val;
        grey_big_img_data[Row * big_width + Col] = (21 * rgba_val.r / 100) + (71 * rgba_val.g /100) + (7  * rgba_val.b /100);
   
    }
}

//Nearest Neighbors but the shared memory, one thread writes to Scale N Pixels
__global__ void nearestNeighbors_shared_memory_Kernel(RGBA_t* big_img_data, unsigned char* grey_big_img_data, RGBA_t* img_data, int big_width, int big_height, int width, int height, int scale)
{
    extern __shared__ RGBA_t img_pixels[];

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int big_x = 0;    int big_y = 0;
    
    RGBA_t rgba_val;

    //BLOCK DIM / SCALE THREADS COLLECT DATA FROM GLOBAL MEMORY
    if (Row < height && Col < width)
    {
        img_pixels[tid_y * blockDim.x + tid_x] = img_data[Row * width + Col];
    }
    //else
    //{
    //    img_pixels[tid_y * blockDim.x + tid_x] = 0;
    //}

    __syncthreads();

    //EVERY (VALID) THREAD PARTICPATES IN OUTPUTTING DATA
    if (Row < height && Col < width)
    {
        for (int y_pix = 0; y_pix < scale; y_pix++)
        {
            for (int x_pix = 0; x_pix < scale; x_pix++)
            {
                big_x = Col * scale + x_pix;
                big_y = Row * scale + y_pix;

                if (Row < big_height && Col < big_width)
                {
                    rgba_val = img_pixels[tid_y * blockDim.x + tid_x];

                    big_img_data[big_y * big_width + big_x] = rgba_val;
                    grey_big_img_data[big_y * big_width + big_x] = (21 * rgba_val.r / 100) + (71 * rgba_val.g / 100) + (7 * rgba_val.b / 100);
                }
            }
        }
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

            //                 R                                   G                                   B
            if((img1[idx + 0] != img2[idx + 0]) || (img1[idx + 1] != img2[idx + 1]) || (img1[idx + 2] != img2[idx + 2]))
            {
                pass = false;
                goto LOOP_EXIT;
            }


        }
    }

LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d\n", x, y);
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
           
            if(img1[idx] != img2[idx])
            {
                pass = false;
                goto GREY_LOOP_EXIT;
            }


        }
    }

GREY_LOOP_EXIT:
    if(!pass)
    {
        printf("Images do not match at pixel X: %d, Y: %d\n", x, y);
    }
    else
    {
        printf("Images match!\n");
    }

}