#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

// RGBA typedef struct
// Struct used to align RGB images to better optimize global memmory access
typedef struct __align__(4) { // Or alignas(4) in C++11
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a; // Padding to ensure 4-byte alignment
}RGBA_t;


// Function to read PPM image
// Parameters
// filename of ppm image to read
// Pass by reference width to set width of read in image
// Pass by reference height to set height of read in image
char* readPPM(char* filename, int* width, int* height) 
{
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

// Function to read gray PPM image
// Parameters
// filename of ppm image to read
// Pass by reference width to set width of read in image
// Pass by reference height to set height of read in image
char* readPPMGray(char* filename, int* width, int* height) 
{
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


// Function to write PPM image
// Parameters
// filename of ppm image to write to
// pointer to rgb data to write
// width of output image
// height of output image
void writePPM(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P6" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height) * 3;

    file.write(img_data, size);
}

// Function to write Gray PPM image
// Parameters
// filename of ppm image to write to
// pointer to rgb data to write
// width of output image
// height of output image
void writePPMGrey(char* filename, char* img_data, int width, int height)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
        throw "File failed to open";

    file << "P5" << "\n" << width << " " << height << "\n" << 255 << "\n";

    size_t size = (width) * (height);

    file.write(img_data, size);
}

#define GREY_CHN    1
#define RGB_CHN     3
// Function to compare images
// Parameters
// Pointer to "true img" data
// Pointer to tested img data
// Number of channesl (GREY) and (RGB)
// Pointers to width and height integers for the image
void Image_Compare(unsigned char* true_img, unsigned char* test_img, int chn_count, int width, int height)
{
    int idx = 0;
    int y;  int x;

    // The largest difference that can occur is 255, there are width * height * channel count of pixels
    float total_pixels = 255.0 * width * height * chn_count;
    float image_difference = 0;

    // For each pixel in the image
    for( y = 0; y < height; y++) 
    {
        for(x = 0; x < width; x++)
        {
            //Calculate the idx
            idx = y * width + x;

            //for each of the channels
            for (int i = 0; i < chn_count; i++)
            {
                //calculate the error of the image
                image_difference += (float)abs((true_img[idx + i] - test_img[idx + i]));
            }
        }
    }
    
    printf("Accuracy: %f%\n", 100 - (100 * image_difference/total_pixels));
}
