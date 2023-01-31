#include<Image.h>
#include<ImageBMP.h>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>

/* .........................................................
*/
Image * read_image_from_file(const char *  file_name) {
  return new ImageBMP(file_name);
}

/* .........................................................
*/
void copy_rgb_image(const ImageRGB * p_source,
					ImageRGB * p_destination) {

  // fprintf (stderr, " vamos a copiar imagen ");

  /* liberar memoria de los pixels p_destination */
  if ((*p_destination).pixels != 0) {
	free ( (*p_destination).pixels );
	(*p_destination).pixels  = 0;
  }

  // fprintf (stderr, " memoria liberada ");

  /* copio todo */
  (*p_destination) = (*p_source); 
  (*p_destination).pixels = 0;

  // fprintf (stderr, " copiados campos ");

  // conseguir memoria para los pixels de la  imagen
  if ((*p_destination).overall_size > 0) {
	(*p_destination).pixels = (unsigned char *) malloc ( (*p_destination).overall_size ); 
	if ( (*p_destination).pixels == 0 ) {
	  fprintf(stderr, " error en malloc() para guardar imagen en memoria \n");
	  throw;
	}

	// copio los bytes
	memcpy 
	  ( (*p_destination).pixels, (*p_source).pixels,  (*p_source).overall_size);
  }

  // fprintf (stderr, " ya he copiado \n");

} /* () */

