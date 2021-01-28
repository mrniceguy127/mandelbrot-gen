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

#define TRUE 1
#define FALSE 0

// Coloring constants - https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
#define K ((double)log10(100000))
#define COLOR_COMON_FACTOR ((double)(1 / log10(2.)))
#define COLOR_R ((double)(1 / (2.5 * sqrt(2.)) * COLOR_COMON_FACTOR))
#define COLOR_G ((double)(1 / (2.4 * sqrt(1.8)) * COLOR_COMON_FACTOR))
#define COLOR_B ((double)(1 * COLOR_COMON_FACTOR))

// Coloring formula.
pixel_t color_x(double x) {
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

void color_mandelbrot_pixmap(zoom_data user_zoom_data, pixmap_t * pixmap, pixel_t COLOR_K) {

  /* Example zoom var vals:
    double zoom_scale = 250000;
    double complex_left = -0.76;
    double complex_bottom = 0.0801; */
  

  // TODO clean.
  // TODO Make coordinate bounds specifiable as arguments to the program

  unsigned int iterates = user_zoom_data.iterates;
  double zoom_scale = user_zoom_data.zoom_scale;
  double complex_width = 3.5/zoom_scale;
  double complex_height = 2.0/zoom_scale;
  double complex_left = user_zoom_data.complex_left;
  double complex_bottom = user_zoom_data.complex_bottom;
  double box[2][2] = {{complex_left,complex_left+complex_width},{complex_bottom,complex_bottom+complex_height}};

  for (unsigned int i = 0; i < pixmap->height; i++) {
    for (unsigned int j = 0; j < pixmap->width; j++) {
      double x_scale_factor = ((complex_width / (double) pixmap->width)); // X scaling multiplier based on pixmanp width and complex coordinate plane height
      double y_scale_factor = ((complex_height / (double) pixmap->height)); // Y scaling based on pixmap height and complex coordinate plane width
      double x = (x_scale_factor * (double) j) + box[0][0]; // Puts x into the complex coordinate plane width bounds and current coordinate position
      double y = (y_scale_factor * (double) i) + box[1][0]; // Puts x into the complex coordinate plane width bounds and current coordinate position
      
      double complex c = x + (I * y); // c from mandelbrot formula
      double complex z = 0.0; // z from mandelbrot formula

      pixel_t * pixel_to_color = pixel_at(pixmap, j, i);
      pixel_t color = COLOR_K;

      double power = 1.;
      unsigned int is_in_mandelbrot = TRUE;
      unsigned int n = 0;

      while (n < iterates && is_in_mandelbrot) {
	double R2 = squared_modulus(z);
        if (R2 > 1000000) {
	  double V = log10(R2)/power;
	  double x = log10(V)/K;
	  color = color_x(x);
	  is_in_mandelbrot = FALSE;
	}
	z = ((z*z) + c);
	power *= 2;
	n++;
      }

      pixel_to_color->r = color.r;
      pixel_to_color->g = color.g;
      pixel_to_color->b = color.b;
    }
  }
}

zoom_data get_zoom_data_from_opts(int argc, char * argv[]) {
  zoom_data user_zoom_data;

  user_zoom_data.zoom_scale = 1.0;
  user_zoom_data.complex_left = -2.5;
  user_zoom_data.complex_bottom = -1.0;
  user_zoom_data.iterates = 100;

  int opt;
  while ((opt = getopt(argc, argv, "i:z:l:b:")) != -1) {
    switch(opt) {
      case 'i':
	user_zoom_data.iterates = atof(optarg);
	break;
      case 'z':
        user_zoom_data.zoom_scale = atof(optarg);
	break;
      case 'l':
	user_zoom_data.complex_left = atof(optarg);
	break;
      case 'b':
	user_zoom_data.complex_bottom = atof(optarg);
	break;
      default:
	break;
    }
  }

  return user_zoom_data;
}

int main(int argc, char * argv[]) {
  const unsigned int IMG_WIDTH = 2801;
  const unsigned int IMG_HEIGHT = 2001;
  
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

  color_mandelbrot_pixmap(user_zoom_data, &pixmap, COLOR_K);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  free(pixmap.pixels);
  if (write_png_status != 0) return -1;
  return 0;
}
