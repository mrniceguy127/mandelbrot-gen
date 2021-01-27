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

void color_mandelbrot_pixmap(pixmap_t * pixmap, pixel_t COLOR_K, unsigned int iterates) {

  /* Example zoom var vals:
    const double zoomScale = 250000;
    double complexLeft = -0.76;
    double complexBottom = 0.0801;
  */

  // TODO clean.

  const double zoomScale = 1;
  double complexWidth = 3.5/zoomScale;
  double complexHeight = 2.0/zoomScale;
  double complexLeft = -2.5;
  double complexBottom = -1.0;
  double box[2][2] = {{complexLeft,complexLeft+complexWidth},{complexBottom,complexBottom+complexHeight}};

  for (unsigned int i = 0; i < pixmap->height; i++) {
    for (unsigned int j = 0; j < pixmap->width; j++) {
      double x_scale_factor = ((complexWidth / (double) pixmap->width)); // X scaling multiplier based on pixmanp width and complex coordinate plane height
      double y_scale_factor = ((complexHeight / (double) pixmap->height)); // Y scaling based on pixmap height and complex coordinate plane width
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


int main() {
  const unsigned int IMG_WIDTH = 2801;
  const unsigned int IMG_HEIGHT = 2001;
 
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

  // TODO Increase iterates for deep zooms. CLI interface would be nice!!
  const unsigned int iterates = 100; 

  color_mandelbrot_pixmap(&pixmap, COLOR_K, iterates);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  if (write_png_status != 0) return -1;
  return 0;
}
