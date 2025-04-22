#pragma once

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

char* readPPM(char* filename, int* width, int* height) 
{
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

char* readPPMGray(char* filename, int* width, int* height) 
{
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