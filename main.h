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

typedef struct {
  unsigned int iterates;
  double zoom_scale;
  double complex_y_offset;
  double complex_x_offset;
} zoom_data;

pixel_t color_x(double x);
double squared_modulus(double complex z);
void color_mandelbrot_pixmap(zoom_data user_zoom_data, pixmap_t * pixmap, pixel_t COLOR_K);
zoom_data get_zoom_data_from_opts(int argc, char * argv[]);

#endif
