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


// TODO add zooming by limiting pixels rendered (w/ i and j) and multiplying the pixmap height and width by some constant
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
static const double K = log10(2);
static const double colorCommonFactor = 1 / log10(2.);
static const double a = 1 * colorCommonFactor;
static const double b = 1 / (3 * sqrt(2.)) * colorCommonFactor;
static const double c = 1 / (7 * pow(3., (1/8))) * colorCommonFactor;

pixel_t color_x(double x) {
  double R = 255 * ((1 - cos(a*x)) / 2);
  double G = 255 * ((1 - cos(b*x)) / 2);
  double B = 255 * ((1 - cos(c*x)) / 2);

  pixel_t color;
  color.r = R;
  color.g = G;
  color.b = B;

  return color;
}

double squared_modulus(complex z) {
  double x = creal(z);
  double y = cimag(z);

  return (x*x) + (y*y);
}

void color_mandelbrot_pixmap(pixmap_t * pixmap, pixel_t COLOR_K, unsigned int iterates) {
  for (unsigned int i = 0; i < pixmap->height; i++) {
    for (unsigned int j = 0; j < pixmap->width; j++) {
      double x_scale_factor = ((3.5 / (double) pixmap->width)); // Keeps x in the scaled range (-2.5+2.5, 1+2.5)
      double y_scale_factor = ((2.0 / (double) pixmap->height)); // Keeps y in the scaled range (-1+1, 1+1)
      double idfk = 2.3; 
      double x = (x_scale_factor * (double) j) - 2.5; // Puts x into the range (-2.5, 1) * pixmap->width
      double y = (y_scale_factor * (double) i) - 1; // Puts y into the range (-1, 1) * pixmap->height
      double complex c = x + (I * y);
      double complex z = 0.0;
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
	z = (z*z) + c;
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

  const unsigned int iterates = 2000;

  color_mandelbrot_pixmap(&pixmap, COLOR_K, iterates);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  if (write_png_status != 0) return -1;
  return 0;
}
