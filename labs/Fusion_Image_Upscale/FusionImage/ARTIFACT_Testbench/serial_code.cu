#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <fstream>
#include <string>

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