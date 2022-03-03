/* main.c - Main mandelbrot generator source file.
*
* Matt Kleiner
* JUNE 2020
*
* gcc v10.1.0
* Linux 5.6.15-arch1-1
*
* Note(s):
*/


// TODO add cli interface
// TODO Parallel processing

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <complex.h>
#include <math.h>
#include <unistd.h>

#include "writepng.h"
#include "main.h"



static const int TRUE = 1;
static const int FALSE = 0;



static const double COMPLEX_X_MIN = -2.0;
static const double COMPLEX_X_MAX = 0.47;

static const double COMPLEX_Y_MIN = -1.12;
static const double COMPLEX_Y_MAX = 1.12;

static const unsigned int IMG_WIDTH = 2470;
static const unsigned int IMG_HEIGHT = 2240;



// Coloring formula.
pixel_t color_x(double x) {
  // Coloring constants - https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
  double k = ((double)log10(100000))
  double colorCommonFactor = ((double)(1 / log10(2.)))
  double color_r = ((double)(1 / (2.5 * sqrt(2.)) * COLOR_COMON_FACTOR))
  double color_g = ((double)(1 / (2.4 * sqrt(1.8)) * COLOR_COMON_FACTOR))
  double color_b = ((double)(1 * COLOR_COMON_FACTOR))

  double R = 255 * ((1 - cos(COLOR_R*x)) / 2);
  double G = 255 * ((1 - cos(COLOR_G*x)) / 2);
  double B = 255 * ((1 - cos(COLOR_B*x)) / 2);

  pixel_t color;
  color.r = R;
  color.g = G;
  color.b = B;

  return color;
}

double squared_modulus(double complex z) {
  double x = creal(z);
  double y = cimag(z);

  return (x*x) + (y*y);
}

double get_scaled_unit(double fake_unit_point, double fake_unit_max, double true_unit_min, double true_unit_max, double zoom_scale) {
  double true_length = true_unit_max - true_unit_min;
  double single_true_unit = (true_length / zoom_scale) / fake_unit_max;
  double true_unit_if_min_is_zero = single_true_unit * fake_unit_point;
  return true_unit_min + true_unit_if_min_is_zero;
}

pixel_t get_mandel_pixel(pixel_t COLOR_K, zoom_data zoomVars, double true_x, double true_y) {
  int max_iterates = zoomVars.iterates;
  int curr_iterate = 0;

  double complex c = true_x + (I * true_y); // c from mandelbrot formula
  double complex z = 0.0; // z from mandelbrot formula

  double power = 1.;
  unsigned int n = 0;
  double R2 = 0;
  double maxR2 = 1000; // creates better coloring when higher.
  
  while (R2 <= maxR2 && curr_iterate < max_iterates) {
    z = ((z*z) + c);
    power *= 2;

    R2 = squared_modulus(z);

    curr_iterate++;
  }

  if (curr_iterate == max_iterates) return COLOR_K;

  double V = log10(R2)/(power);
  double x = log10(V)/K;
  return color_x(x);
}

void draw(pixel_t COLOR_K, zoom_data zoomVars, pixmap_t * screen) {
  int px;
  int py;
  for (px = 0; px < screen->width; px++) {
    for (py = 0; py < screen->height; py++) {
      double complex_x_offset = zoomVars.complex_x_offset;
      double complex_y_offset = zoomVars.complex_y_offset;
      double true_x = get_scaled_unit((double) px, (double) screen->width, COMPLEX_X_MIN + complex_x_offset, COMPLEX_X_MAX + complex_x_offset, zoomVars.zoom_scale);
      double true_y = get_scaled_unit((double) py, (double) screen->height, COMPLEX_Y_MIN + complex_y_offset, COMPLEX_Y_MAX + complex_y_offset, zoomVars.zoom_scale);
      pixel_t color_to_draw = get_mandel_pixel(COLOR_K, zoomVars, true_x, true_y);
      screen->pixels[(py * (screen->width)) + px] = color_to_draw;
    }
  }
}



zoom_data get_zoom_data_from_opts(int argc, char * argv[]) {
  zoom_data user_zoom_data;

  user_zoom_data.zoom_scale = 1.0;
  user_zoom_data.complex_x_offset = 0;
  user_zoom_data.complex_y_offset = 0;
  user_zoom_data.iterates = 100;

  int opt;
  while ((opt = getopt(argc, argv, "i:z:x:y:")) != -1) {
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
 
  pixmap_t pixmap;
  pixmap.pixels = calloc(IMG_WIDTH * IMG_HEIGHT, sizeof(pixel_t));
  pixmap.width  = IMG_WIDTH;
  pixmap.height = IMG_HEIGHT;

  if (!pixmap.pixels) {
    return -1;
  }
  
  pixel_t COLOR_K; // 'Special' color (in the mandelbrot, so black probably)
  COLOR_K.r = 0x00;
  COLOR_K.g = 0x00;
  COLOR_K.b = 0x00;

  draw(COLOR_K, user_zoom_data, &pixmap);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  free(pixmap.pixels);
  if (write_png_status != 0) return -1;
  return 0;
}
