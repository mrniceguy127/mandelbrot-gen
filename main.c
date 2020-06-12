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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <complex.h>

#include "writepng.h"
#include "main.h"

#define TRUE 1
#define FALSE 0

unsigned int squared_modulus(complex z) {
  unsigned int x = creal(z);
  unsigned int y = cimag(z);

  return (x*x) + (y*y);
}

void color_mandelbrot_pixmap(pixmap_t * pixmap, pixel_t COLOR_K, pixel_t COLOR_NOTK, unsigned int iterates) {
  for (unsigned int i = 0; i < pixmap->height; i++) {
    for (unsigned int j = 0; j < pixmap->width; j++) {
      float scaledJ = (2.5 / (float) pixmap->width);
      float scaledI = (2 / (float) pixmap->height);
      float complex c = ((scaledJ * (float) j) - 2) + (((scaledI * (float) i) - 1) * I);
      float complex z = 0.0;
      pixel_t * pixel_to_color = pixel_at(pixmap, j, i);
      unsigned int n = 0;
      int is_in_mandelbrot = TRUE;
      while (n < iterates && is_in_mandelbrot) {
        if (squared_modulus(z) > 4) {
	  pixel_to_color->r = COLOR_NOTK.r;
	  pixel_to_color->g = COLOR_NOTK.g;
	  pixel_to_color->b = COLOR_NOTK.b;
	  is_in_mandelbrot = FALSE;
	}
	z = (z*z) + c;
	n++;
      }
      if (is_in_mandelbrot) {
        pixel_to_color->r = COLOR_K.r;
        pixel_to_color->g = COLOR_K.g;
        pixel_to_color->b = COLOR_K.b;
      }
    }
  }
}


int main() {
  const unsigned int IMG_WIDTH = 1000;
  const unsigned int IMG_HEIGHT = 1000;
 
  pixmap_t pixmap;
  pixmap.pixels = calloc(IMG_WIDTH * IMG_HEIGHT, sizeof(pixel_t));
  pixmap.width  = IMG_WIDTH;
  pixmap.height = IMG_HEIGHT;

  if (!pixmap.pixels) {
    return -1;
  }
  
  int status; 
  status = 0;

  pixel_t COLOR_NOTK; 
  COLOR_NOTK.r = 0x0;
  COLOR_NOTK.g = 0x0;
  COLOR_NOTK.b = 0x0;
  pixel_t COLOR_K; // 'Special' color
  COLOR_K.r = 0xFF;
  COLOR_K.g = 0xFF;
  COLOR_K.b = 0xFF;

  const unsigned int iterates = 1000;

  color_mandelbrot_pixmap(&pixmap, COLOR_K, COLOR_NOTK, iterates);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  if (write_png_status != 0) return -1;
  return 0;
}
