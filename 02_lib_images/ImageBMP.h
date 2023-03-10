// ---------------------------------------------------------
// ---------------------------------------------------------
/* ImageBMP.h */
// ---------------------------------------------------------
// ---------------------------------------------------------

#ifndef IMAGE_BMP_H
#define IMAGE_BMP_H

#include<Image.h>

/* .........................................................
*/
#pragma pack(2) /*2 byte packing */ 
typedef struct { 
  unsigned short int type; 
  unsigned int size; 
  unsigned short int reserved1,reserved2; 
  unsigned int offset; 
} BMP_File_Header1;

/* .........................................................
*/
#pragma pack() /* Default packing */ 
typedef struct { 
  unsigned int size;  // 4
  int width; // 4
  int height;  // 4
  unsigned short int planes;  // 2
  unsigned short int bits_per_pixel;  // 2
  unsigned int compression;  // 4
  unsigned int imagesize;  // 4
  int xresolution,yresolution; 
  unsigned int ncolors;  // 4
  unsigned int importantcolors;  // 4
} BMP_File_Header2;

/* .........................................................
*/

/* .........................................................
*/
class ImageBMP : public Image 
{
 private:
  BMP_File_Header1 header1;
  BMP_File_Header2 header2;
  void free_pixels();
  static const PixelRGB white_pixel; /* constante pero se incializa en el .cpp */

 public:

  //ImageRGB  the_image; inherited from class Image 

  ImageBMP(const char * file_name) ;

  ImageBMP(
		   const unsigned int height_rows, 
		   const unsigned int width_columns,
		   PixelRGB color = white_pixel);

  /* access a pixel by row, column */
  PixelRGB * pixel (const unsigned int row, const unsigned int col);

  void save_to_file(const char * file_name) const;

  void print_information() const;

  void operator=(const ImageBMP &);

  ~ImageBMP();
}; /* class */

/* .........................................................
 */
void copy_adding_padding_3to4 (const unsigned char * p_source, 
							   unsigned char * p_destination,
							   unsigned int tamanyo);

void copy_removing_padding_4to3(
								const unsigned char * p_source, 
								unsigned char * p_destination,
								unsigned int size);

#endif 

// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
