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

pixel_t color_n(unsigned int n);
float squared_modulus(complex z);
void color_mandelbrot_pixmap(pixmap_t * pixmap, pixel_t COLOR_K, unsigned int iterates);

#endif
