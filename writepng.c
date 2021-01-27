/* writepng.c - Source file for writing pixmaps created in main.c to a png.
*
* Matt Kleiner
* JUNE 2020
*
* gcc v10.1.0
* Linux 5.6.15-arch1-1
*
* Note(s):
*/

#include <stdint.h>
#include <stdlib.h>

#include <png.h>

#include "writepng.h"

pixel_t * pixel_at(pixmap_t * pixmap, unsigned int x, unsigned int y) {
  return pixmap->pixels + (pixmap->width * y) + x;
}

int write_png_from_pixmap(pixmap_t * pixmap, const char * path) {
  const unsigned int pixel_size = 3; // size of pixel in bytes.
  const unsigned int depth = 8; // Pixel bit depth.

  FILE * fp = fopen(path, "wb");
  if (!fp) return -1;

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) return -1;

  png_infop info = png_create_info_struct(png);
  if (!info) return -1;

  if (setjmp(png_jmpbuf(png))) return -1;

  // Out has 8bit depth in RGB format.
  png_set_IHDR(
    png,
    info,
    pixmap->width,
    pixmap->height,
    depth,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  png_byte ** row_pointers = png_malloc(png, pixmap->height * sizeof(png_byte *));
  
  for (unsigned int y = 0; y < pixmap->height; y++) {
    png_byte * row = png_malloc(png, sizeof(uint8_t) * pixmap->width * pixel_size);
    row_pointers[y] = row;
    for (unsigned int x = 0; x < pixmap->width; x++) {
      pixel_t * pixel = pixel_at(pixmap, x, y);
      *row++ = pixel->r;
      *row++ = pixel->g;
      *row++ = pixel->b;
    }
  }

  png_init_io(png, fp);
  png_set_rows(png, info, row_pointers);
  png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

  for (unsigned int y = 0; y < pixmap->height; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);

  fclose(fp);

  png_destroy_write_struct(&png, &info);

  return 0;
}
