#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <unistd.h>

#include "writepng.h"
#include "main.h"




static const double COMPLEX_X_MIN = -2.0;
static const double COMPLEX_X_MAX = 0.47;

static const double COMPLEX_Y_MIN = -1.12;
static const double COMPLEX_Y_MAX = 1.12;

static const unsigned int IMG_WIDTH = 2470;
static const unsigned int IMG_HEIGHT = 2240;

static const double K_LOG_POWER = 100000; // K = log10(K_LOG_POWER) -  https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set




// Coloring formula.
__device__
pixel_t color_x(double x) {
  // Coloring constants - https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
  double k = log10(K_LOG_POWER);
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

__device__
double squared_modulus(complex z) {
  double x = z.real;
  double y = z.imag;

  return (x*x) + (y*y);
}

__device__
double get_scaled_unit(double fake_unit_point, double fake_unit_max, double true_unit_min, double true_unit_max, double zoom_scale) {
  double true_length = true_unit_max - true_unit_min;
  double single_true_unit = (true_length / zoom_scale) / fake_unit_max;
  double true_unit_if_min_is_zero = single_true_unit * fake_unit_point;
  return true_unit_min + true_unit_if_min_is_zero;
}

__device__
complex mulComplex(complex x, complex y) {
  double a = x.real;
  double b = x.imag;
  double c = y.real;
  double d = y.imag;

  // FOIL
  double f = a * c;
  double o = a * d;
  double i = b * c;
  double l = b * d;

  double real = f - l;
  double imag = o + i;

  complex result;
  result.real = real;
  result.imag = imag;

  return result;
}
__device__
complex addComplex(complex x, complex y) {
  complex result;
  result.real = x.real + y.real;
  result.imag = x.imag + y.imag;
  return result;
}


__device__
pixel_t get_mandel_pixel(zoom_data zoomVars, double true_x, double true_y) {
  int max_iterates = zoomVars.iterates;
  int curr_iterate = 0;

  complex c;
  c.real = true_x;
  c.imag = true_y; // c from mandelbrot formula
 
  complex z;
  z.real = 0.0; // z from mandelbrot formula
  z.imag = 0.0;

  double power = 1.0;
  double R2 = 0;
  double maxR2 = 1000; // creates better coloring when higher.
  
  while (R2 <= maxR2 && curr_iterate < max_iterates) {
    z = addComplex(mulComplex(z, z), c);
    power *= 2;

    R2 = squared_modulus(z);

    curr_iterate++;
  }


  if (curr_iterate == max_iterates) {
    pixel_t COLOR_K; // 'Special' color (in the mandelbrot, so black probably)
    COLOR_K.r = 0x00;
    COLOR_K.g = 0x00;
    COLOR_K.b = 0x00;
    return COLOR_K;
  }

  double V = log10(R2)/(power);
  double x = log10(V)/log10(K_LOG_POWER);
  return color_x(x);
}

__device__
void drawChunk(zoom_data zoomVars, pixmap_t * screen, unsigned int chunkIdx) {
  int px = chunkIdx % IMG_WIDTH;
  int py = (chunkIdx - px) / IMG_WIDTH;

  double complex_x_offset = zoomVars.complex_x_offset;
  double complex_y_offset = zoomVars.complex_y_offset;
  double true_x = get_scaled_unit((double) px, (double) screen->width, COMPLEX_X_MIN + complex_x_offset, COMPLEX_X_MAX + complex_x_offset, zoomVars.zoom_scale);
  double true_y = get_scaled_unit((double) py, (double) screen->height, COMPLEX_Y_MIN + complex_y_offset, COMPLEX_Y_MAX + complex_y_offset, zoomVars.zoom_scale);
  screen->pixels[(py * (screen->width)) + px] = get_mandel_pixel(zoomVars, true_x, true_y);
}

__global__
void draw(int n, zoom_data zoomVars, pixmap_t * screen) {
  int chunkIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int i;
  for (i = chunkIdx; i < n; i += stride) {
    drawChunk(zoomVars, screen, i);
  }
}



zoom_data get_zoom_data_from_opts(int argc, char * argv[]) {
  zoom_data user_zoom_data;

  user_zoom_data.zoom_scale = 1.0;
  user_zoom_data.complex_x_offset = 0;
  user_zoom_data.complex_y_offset = 0;
  user_zoom_data.iterates = 100;

  int opt;
  while ((opt = getopt(argc, argv, "i:z:x:y:t:")) != -1) {
    switch(opt) {
      case 'i':
        user_zoom_data.iterates = atof(optarg);
        break;
      case 'z':
        user_zoom_data.zoom_scale = atof(optarg);
	break;
      case 'y':
        user_zoom_data.complex_y_offset = atof(optarg);
        break;
      case 'x':
	user_zoom_data.complex_x_offset = atof(optarg);
	break;
      default:
	break;
    }
  }

  return user_zoom_data;
}


int main(int argc, char * argv[]) {
  zoom_data user_zoom_data = get_zoom_data_from_opts(argc, argv);
 
  int N = IMG_WIDTH * IMG_HEIGHT;

  pixmap_t * pixmap;
  cudaMallocManaged(&pixmap, sizeof(pixmap));
  cudaMallocManaged(&pixmap->pixels, N * sizeof(pixel_t));
  pixmap->width  = IMG_WIDTH;
  pixmap->height = IMG_HEIGHT;

  if (!pixmap->pixels) {
    return -1;
  }
 

  int blockSize = 768;
  int numBlocks = (N + blockSize - 1) / blockSize;
  draw<<<numBlocks, blockSize>>>(N, user_zoom_data, pixmap);
  cudaDeviceSynchronize();

  unsigned int write_png_status = write_png_from_pixmap(pixmap, "mandel.png");

  cudaFree(&pixmap->pixels);

  if (write_png_status != 0) {
    return -1;
  }

  return 0;
}
