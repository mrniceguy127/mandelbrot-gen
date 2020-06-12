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

pixel_t color_n(unsigned int n) {
  unsigned int i = n % 16;

  pixel_t col0; col0.r = 66; col0.g = 30; col0.b = 15;
  pixel_t col1; col1.r = 25; col1.g = 7; col1.b = 26;
  pixel_t col2; col2.r = 9; col2.g = 1; col2.b = 47;
  pixel_t col3; col3.r = 4; col3.g = 4; col3.b = 73;
  pixel_t col4; col4.r = 0; col4.g = 7; col4.b = 100;
  pixel_t col5; col5.r = 12; col5.g = 44; col5.b = 138;
  pixel_t col6; col6.r = 24; col6.g = 82; col6.b = 177;
  pixel_t col7; col7.r = 57; col7.g = 125; col7.b = 209;
  pixel_t col8; col8.r = 134; col8.g = 181; col8.b = 229;
  pixel_t col9; col9.r = 211; col9.g = 236; col9.b = 248;
  pixel_t col10; col10.r = 241; col10.g = 233; col10.b = 191;
  pixel_t col11; col11.r = 248; col11.g = 201; col11.b = 95;
  pixel_t col12; col12.r = 255; col12.g = 170; col12.b = 0;
  pixel_t col13; col13.r = 204; col13.g = 128; col13.b = 0;
  pixel_t col14; col14.r = 153; col14.g = 87; col14.b = 0;
  pixel_t col15; col15.r = 106; col15.g = 52; col15.b = 3;

  pixel_t cols[] = { col0, col1, col2, col3, col4, col5, col6, col7,
                     col8, col9, col10, col11, col12, col13, col14, col15 };

  return cols[i];
}

float squared_modulus(complex z) {
  float x = creal(z);
  float y = cimag(z);

  return (x*x) + (y*y);
}

void color_mandelbrot_pixmap(pixmap_t * pixmap, pixel_t COLOR_K, unsigned int iterates) {
  for (unsigned int i = 0; i < pixmap->height; i++) {
    for (unsigned int j = 0; j < pixmap->width; j++) {
      float x_scale_factor = (3.5 / (float) pixmap->width); // Keeps x in the scaled range (-2.5+2.5, 1+2.5)
      float y_scale_factor = (2.0 / (float) pixmap->height); // Keeps y in the scaled range (-1+1, 1+1)
      float x = (x_scale_factor * (float) j) - 2.5; // Puts x into the range (-2.5, 1) * pixmap->width
      float y = (y_scale_factor * (float) i) - 1; // Puts y into the range (-1, 1) * pixmap->height
      float complex c = x + (I * y);
      float complex z = 0.0;
      pixel_t * pixel_to_color = pixel_at(pixmap, j, i);

      pixel_t color = COLOR_K;

      unsigned int n = 0;
      unsigned int is_in_mandelbrot = TRUE;
      while (n < iterates && is_in_mandelbrot) {
        if (squared_modulus(z) > 4) {
	  color = color_n(n);
	  is_in_mandelbrot = FALSE;
	}
	z = (z*z) + c;
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

  const unsigned int iterates = 500;

  color_mandelbrot_pixmap(&pixmap, COLOR_K, iterates);
  unsigned int write_png_status = write_png_from_pixmap(&pixmap, "mandel.png");
  
  if (write_png_status != 0) return -1;
  return 0;
}
