/**
 * @file main.cu
 * @author Matt Kleiner (no@spam.com)
 * @brief The main program for drawing the mandelbrot set using NVidia's CUDA toolkit
 * @version 0.1
 * @date 2022-10-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */





#include <iostream>

#include <unistd.h>

#include "writepng.h"
#include "main.h"







/**
 * Convoluted coloring formula.
 * 
 * @param x used to determine the color
 * @param bailoutRadius the bailout radius for this run. CAN be useful for coloring.
 * @return pixel to draw 
 */
__device__
pixel_t colorX(double x, double bailoutRadius) {
  // Coloring constants - https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
  double k = log10(bailoutRadius);
  double colorCommonFactor = (double) (1 / log10(2.0));
  double color_r = (double) (1 / (2.5 * sqrt(2.0)) * colorCommonFactor);
  double color_g = (double) (1 / (2.4 * sqrt(1.8)) * colorCommonFactor);
  double color_b = (double) (1 * colorCommonFactor);

  pixel_t color;
  color.r = 255 * ((1 - cos(color_r*x)) / 2);
  color.g = 255 * ((1 - cos(color_g*x)) / 2);
  color.b = 255 * ((1 - cos(color_b*x)) / 2);

  return color;
}

/**
 * Get complex coordinate based on point on a number line (e.g. mandel coordinate based on pixel in a column)
 *
 * @param fakeUnitPoint A point in our "fake" coordinate axis (e.g. a pixel in a column)
 * @param fakeUnitMax The maximum value of the fake coordinate axis (e.g. pixel column height)
 * @param trueUnitMin The minimum value we want to draw on the mandelbrot set on a particular axis
 * @param trueUnitMax The maximum value we want to draw on the mandelbrot set on a particular axis
 * @return double representing the coordinate on an axis of the mandelbrot plane based on a pixel
 */
__device__
double getScaledUnit(double fakeUnitPoint, double fakeUnitMax, double trueUnitMin, double trueUnitMax) {
  double trueLength = trueUnitMax - trueUnitMin;
  double singleTrueUnit = trueLength / fakeUnitMax;
  double trueUnitOffset = singleTrueUnit * fakeUnitPoint;
  return trueUnitMin + trueUnitOffset;
}


/**
 * @brief Get the Mandel Pixel object using periodicity checking. Memory expensive, but faster.
 * 
 * @param trueX mandelbrot x coordinate to get colot for
 * @param trueY mandelbrot y coordinate to get colot for
 * @param zoomData various information for this zoom
 * @return pixel_t the pixel color to draw for these coordinates 
 */
__device__
pixel_t getMandelPixel(double trueX, double trueY, zoomdata_t& zoomData) {
  int maxIterates = zoomData.maxIterates;
  int currIterate = 0;

  double bailoutRadius = 1000000; // K = log10(bailoutRadius) -  https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set




  // z real (x) and z imag (y) complex num from mandelbrot formula
  double x = 0.0;
  double y = 0.0;

  double xSquared = 0;
  double ySquared = 0;

  double R2 = 0;

  double xOld = 0;
  double yOld = 0;
  double period = 0;
  
  while (R2 <= bailoutRadius && currIterate < maxIterates) {
    y = 2 * x * y + trueY;
    x = xSquared - ySquared + trueX;

    xSquared = x*x;
    ySquared = y*y;

    R2 = xSquared + ySquared; // Squared modulus

    currIterate++;

    if (x == xOld && y == yOld) {
      currIterate = maxIterates;
    } else {
      period++;
      if (period > 20) {
        period = 0;
        xOld = x;
        yOld = y;
      }
    }
  }


  if (currIterate == maxIterates) {
    pixel_t COLOR_K; // 'Special' color (in the mandelbrot, so black probably)
    COLOR_K.r = 0x00;
    COLOR_K.g = 0x00;
    COLOR_K.b = 0x00;
    return COLOR_K;
  }


  double V = log10(R2) / (pow(2, currIterate));
  double colX = log10(V) / log10(bailoutRadius);
  pixel_t mandelPixel = colorX(colX, bailoutRadius);
  return mandelPixel;
}

/**
 * @brief draw a "chunk." In other words, draw the amount of pixels that each thread is supposed to draw.
 * 
 * @param chunkIdx the chunk index to draw.
 * @param screen The pixmap for the image we're drawing
 * @param zoomData Various info about the current zoom.
 */
__device__
void drawChunk(unsigned int chunkIdx, pixmap_t& screen, zoomdata_t& zoomData) {
  unsigned int height = screen.height;
  unsigned int width = screen.width;

  int px = chunkIdx % width;
  int py = (int) (chunkIdx / width);

  double trueX = getScaledUnit((double) px, (double) width, zoomData.mandelX0, zoomData.mandelX1);
  double trueY = getScaledUnit((double) py, (double) height, zoomData.mandelY0, zoomData.mandelY1);
  pixel_t mandelPixel = getMandelPixel(trueX, trueY, zoomData);
  screen.pixels[((width * height) - 1) - ((py * (width)) + (width - px))] = mandelPixel;
}

/**
 * @brief draw the mandelbrot set. Uses CUDA.
 * 
 * @param n Number of pixels to draw
 * @param screen The pixmap for the image we're drawing
 * @param zoomData Various info about the current zoom.
 * @return __global__ 
 */
__global__
void draw(int n, pixmap_t& screen, zoomdata_t& zoomData) {
  int chunkIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int i;
  for (i = chunkIdx; i < n; i += stride) {
    drawChunk(i, screen, zoomData);
  }
}


/**
 * @brief Process and store information for the zoom from the users provided command line options.
 * 
 * @param argc argc from main
 * @param argv argv from main
 * @param zoomData Various info about the current zoom.
 * @param screen The pixmap for the image we're drawing
 */
void processCLIOpts(int argc, char * argv[], zoomdata_t& zoomData, pixmap_t& screen) {

  zoomData.maxIterates = 100;
  zoomData.mandelX0 = -2;
  zoomData.mandelX1 = 1;
  zoomData.mandelY0 = -1.5;
  zoomData.mandelY1 = 1.5;
  screen.width = 1500;

  int opt;
  while ((opt = getopt(argc, argv, "i:x:y:n:m:w:")) != -1) {
    switch(opt) {
      case 'i':
        zoomData.maxIterates = std::stod(optarg);
        break;
      case 'y':
        zoomData.mandelY0 = std::stod(optarg);
        break;
      case 'm':
        zoomData.mandelY1 = std::stod(optarg);
        break;
      case 'x':
        zoomData.mandelX0 = std::stod(optarg);
        break;
      case 'n':
        zoomData.mandelX1 = std::stod(optarg);
        break;
      case 'w':
        screen.width = std::stoi(optarg);
        break;
      default:
        break;
    }
  }
}


/**
 * @brief Main program for drawing the mandelbrot set using NVidia's CUDA toolkit. 
 * 
 * @param argc Number of arguments to the program
 * @param argv The arguments to the program themselves
 * @return Exit code
 */
int main(int argc, char * argv[]) {
  zoomdata_t * zoomData;
  cudaMallocManaged(&zoomData, sizeof(zoomData));

  pixmap_t * pixmap;
  cudaMallocManaged(&pixmap, sizeof(pixmap));

  processCLIOpts(argc, argv, *zoomData, *pixmap);
  printf("%f, %f, %f, %f\n", zoomData->mandelX0, zoomData->mandelX1, zoomData->mandelY0, zoomData->mandelY1);
  pixmap->height = (int) (pixmap->width * std::abs(( (zoomData->mandelY1 - zoomData->mandelY0) / (zoomData->mandelX1 - zoomData->mandelX0) )));
  printf("%d, %d\n", pixmap->width, pixmap->height);

  int N = pixmap->width * pixmap->height;
  cudaMallocManaged(&pixmap->pixels, N * sizeof(pixel_t));

  if (!pixmap->pixels) {
    return -1;
  }

  int blockSize = 1024; // CUDA Compute Capability 3.5+
  int numBlocks = (N + blockSize - 1) / blockSize;
  draw<<<numBlocks, blockSize>>>(N, *pixmap, *zoomData);
  cudaDeviceSynchronize();

  unsigned int write_png_status = write_png_from_pixmap(pixmap, "mandel.png");

  cudaFree(&pixmap->pixels);

  if (write_png_status != 0) {
    return -1;
  }

  return 0;
}
