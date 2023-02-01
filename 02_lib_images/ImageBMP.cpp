// ---------------------------------------------------------
// ---------------------------------------------------------
/* ImageBMP.cpp */
// ---------------------------------------------------------
// ---------------------------------------------------------

#include <ImageBMP.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* .........................................................
*/
/*
Image * read_bmp_image_from_file(const char *  file_name) {
  return new ImageBMP(file_name);
}
*/

/* .........................................................
  constructor
*/
ImageBMP::ImageBMP(const char *  file_name) {
  FILE *file_ref;

  /* 
   * 
   */
  the_image.pixels = 0;

  /* abro el file_refero */
  file_ref = fopen(file_name,"r"); 
  if (file_ref == 0) {
	free_pixels();
	throw Error_FileNotFound();
  }

  /* read headers */
  fread( & header1,sizeof(BMP_File_Header1),1,file_ref); 
  fread( & header2,sizeof(BMP_File_Header2),1,file_ref); 

  /* find out width, height, bits x pixel */
  the_image.width = header2.width;
  the_image.height = header2.height;

  the_image.overall_size = header2.imagesize;
  
  the_image.bytes_per_row = the_image.overall_size / the_image.height;

  the_image.bits_per_pixel = header2.bits_per_pixel; 

  /* read the pixels */
  /* set the position where they start */

  if ( fseek (file_ref, header1.offset,SEEK_SET) != 0 ) {
	/* printf (" error in fseek \n"); */
	free_pixels();
	throw Error_ReadingFile{};
  }

  unsigned long int n_bytes_to_read = the_image.overall_size;

  /* get memory for the bytes */
  the_image.pixels = (unsigned char *) malloc ( n_bytes_to_read ); 
  if ( the_image.pixels == 0 ) {
	free_pixels();
	printf( " error en malloc() para guardar imagen en memoria \n");
	throw;
  }

  /* read the bytes */
  unsigned long int n_read;
  n_read = fread(the_image.pixels, 1, n_bytes_to_read, file_ref); 
  if ( n_read != n_bytes_to_read ) {
	free_pixels();
	throw Error_ReadingFile();
  }

  /* close the file */
  fclose (file_ref);
} // ()

/* .........................................................
*/
const PixelRGB ImageBMP::white_pixel = {255,255,255};

/* .........................................................
  constructor
*/
ImageBMP::ImageBMP(
				   const unsigned int height, // num. of rows
				   const unsigned int width, // num. of width
				   PixelRGB color) {

  // fill in the fields of ImageRGB
  the_image.height = height;
  the_image.width = width;
  the_image.bits_per_pixel = 24; // 24 bits = 8 bytes <= RGB
  the_image.bytes_per_row = (width * 3);
  the_image.bytes_per_row += (4-(the_image.bytes_per_row%4))%4; // anyadir padding
  the_image.pixels = 0;
  the_image.overall_size = the_image.bytes_per_row * the_image.height;

  // get memory for the pixels 
  if (the_image.overall_size > 0) {
	the_image.pixels = (unsigned char *) malloc ( the_image.overall_size ); 
	if ( the_image.pixels == nullptr ) {
	  free_pixels();
	  throw Error_Malloc();
	}
	// set the pixels to the color given
	PixelRGB * pAux = (PixelRGB*) the_image.pixels;
	for (int i = 0; i<= (the_image.height*the_image.width)-1; i++) {
	  pAux[i] = color;
	}
  }

  // fill in the headers of file .bmp

  /* header 1 */
  header1.type = 19778;   /* initial chars of a bmp file? */
  header1.offset = 54;  /* size of the two headers altogehter */
  header1.size = the_image.overall_size + header1.offset;
  /* set reserved fields to */
  header1.reserved1 = 0;
  header1.reserved2 = 0;

  /* header 2 */
  header2.size = 40; /* size in bytes of this header */
  header2.width = the_image.width;
  header2.height = the_image.height; 
  header2.planes = 1;
  header2.bits_per_pixel = the_image.bits_per_pixel;
  header2.compression = 0;
  header2.imagesize = the_image.overall_size; 
  header2.xresolution = 2835;
  header2.yresolution = 2835;
  header2.ncolors = 0;
  header2.importantcolors = 0;

}  // constructor

/* .........................................................
 */
void ImageBMP::free_pixels() {
  if (the_image.pixels == 0) {
	return;
  }
  free ( the_image.pixels );
  the_image.pixels  = 0;
} // ()

/* .........................................................
 */
ImageBMP::~ImageBMP() {
  free_pixels();
} // ()

/* .........................................................
 */
void ImageBMP::save_to_file(const char * file_name) const {
  FILE * fout;
  fout = fopen(file_name,"w"); 

  if (fout == 0) {
	return;
  }

  fwrite( & header1, sizeof(BMP_File_Header1),1,fout); 
  fwrite( & header2, sizeof(BMP_File_Header2),1,fout); 
  fwrite( the_image.pixels, 1 , header2.imagesize, fout);

  fclose (fout);
} // ()

/* .........................................................
 */
void ImageBMP::print_information() const {
  printf (" ImageRGB ---------\n ");
  printf (" height=%d width=%d bytes_per_row=%d bytes_per_row (without pad) =%d overall_size (without pad pixel=3 bytes)=%d\n",
		  the_image.height,
		  the_image.width,
		  the_image.bytes_per_row,
		  the_image.width*3,
		  the_image.overall_size
		  );

  printf (" header header 1 ---------\n ");
  printf (" type=%d, size=%d, offset=%d\n", 
		  header1.type,
		  header1.size,
		  header1.offset);
  printf (" header header 2 ---------\n ");
  printf (" size=%d, width=%d, heigth=%d, planes=%d, bits_per_pixel=%d, compression=%d, imagesize=%d, xresolution=%d, yresolution=%d, ncolors=%d, importantcolors=%d\n", 
		  header2.size,
		  header2.width,
		  header2.height,
		  header2.planes,
		  header2.bits_per_pixel,
		  header2.compression,
		  header2.imagesize,
		  header2.xresolution,
		  header2.yresolution,
		  header2.ncolors,
		  header2.importantcolors);
} // ()

/* .........................................................
   copy with =
*/
void ImageBMP::operator=(const ImageBMP & other) {


  free_pixels();

  /* copy headers*/

  header1 = other.header1;
  header2 = other.header2;

  /* copy image including pixels */
  copy_rgb_image(& other.the_image,  &the_image);

} // ()

/* .........................................................
   access a pixel 
*/
PixelRGB * ImageBMP::pixel (const unsigned int row, const unsigned int col) {
  return get_pixel ( &(*this).the_image, row, col);
} // ()

/* .........................................................
 */
void copy_adding_padding_3to4 (
							   const unsigned char * p_source, 
							   unsigned char * p_destination,
							   unsigned int size) {
  unsigned int j=0;
  for (unsigned int i=0; i<size; i++) {
	p_destination[j] = p_source[i];
	if (i%3 == 2) {
	  // each time the 3rd byte (index 2) is copied, add a fourth (value = 0) 
	  j++;
	  p_destination[j] = 0;
	} // if

	j++;
  } // for
} // ()

/* .........................................................
 */
void copy_removing_padding_4to3 (
								 const unsigned char * p_source, 
								 unsigned char * p_destination,
								 unsigned int size) {
  
  unsigned int j=0;
  for (unsigned int i=0; i<=size-1; i++) {
	if ( (i%4) < 3 ) {
	  // copy only bytes 0 1 and 2, not the fourth
	  p_destination[j] = p_source[i];
	  j++;
	}
  } // for
} // ()

// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
