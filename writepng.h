/* writepng.h - Header file for writing pixmaps created in main.c to a png.
*
* Matt Kleiner
* JUNE 2020
*
* gcc v10.1.0
* Linux 5.6.15-arch1-1
*
* Note(s):
*/

#ifndef WRITEPNG_H
#define WRITEPNG_H

typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} pixel_t;

typedef struct {
  pixel_t * pixels;
  size_t width;
  size_t height;
} pixmap_t;

pixel_t * pixel_at (pixmap_t * pixmap, unsigned int x, unsigned int y);
int write_png_from_pixmap(pixmap_t * pixmap, const char * path);

#endif
