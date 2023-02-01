// ---------------------------------------------------------
/* Image.h */
// ---------------------------------------------------------

#ifndef IMAGE_H
#define IMAGE_H

#include <util.h>

/* .........................................................
*/
typedef struct ImageRGB {
 public:
  unsigned int width;
  unsigned int height;
  unsigned int bits_per_pixel;  /* 1 2 8 24(RGB)  32(RGB+alpha) */
  unsigned char * pixels; /* array bytes */
  unsigned int bytes_per_row;
  unsigned int overall_size;  // 4
} ImageRGB;

/* para acceder a un pixel dado un puntero al anterior
 * struct y (f,c) */
#define get_pixel(imPtr, f,c) \
  (PixelRGB*) & (*(imPtr)).pixels[((f)*(*imPtr).bytes_per_row)+(3*(c))]

void copy_rgb_image(const ImageRGB * p_source,
					ImageRGB * p_destinatin);

/* .........................................................
typedef struct {
  unsigned int row;
  unsigned int column;
} Coordinate;
*/

/* .........................................................
*/
typedef struct {
  unsigned char b;
  unsigned char g; /*  BGR actually */
  unsigned char r;
} PixelRGB;

/* .........................................................
*/
typedef struct {
  unsigned char b;
  unsigned char g; /* caution: it is BGR actually !!! */
  unsigned char r;
  unsigned char a;
} PixelRGBA;

/* .........................................................
*/
typedef struct {
  unsigned char b;
  unsigned char g;
  unsigned char r;
} PixelBGR;

/* .........................................................
*/
class Image {
 public:
  ImageRGB * the_image;
  virtual void save_to_file(const char * file_name) const = 0;
  virtual ~Image() { }
}; /* class */

#endif 

// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
