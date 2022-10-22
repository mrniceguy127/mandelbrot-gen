/* main.h - Main mandelbrot generator header file.
*
* Matt Kleiner
* JUNE 2020
*
* gcc v10.1.0
* Linux 5.6.15-arch1-1
*
* Note(s):
*/

#ifndef MAIN_H
#define MAIN_H

struct zoomdata_t {
    double mandelX0;
    double mandelX1;
    double mandelY0;
    double mandelY1;
    unsigned long int maxIterates;
};


__device__
pixel_t colorX(double x, double bailoutRadius);

__device__
double getScaledUnit(double fakeUnitPoint, double fakeUnitMax, double trueUnitMin, double trueUnitMax);

__device__
pixel_t getMandelPixel(double trueX, double trueY, zoomdata_t& zoomData);

__device__
void drawChunk(unsigned int chunkIdx, pixmap_t& screen, zoomdata_t& zoomData);

__global__
void draw(int n, pixmap_t& screen, zoomdata_t& zoomData);

void processCLIOpts(int argc, char * argv[], zoomdata_t& zoomData, pixmap_t& screen);


#endif
