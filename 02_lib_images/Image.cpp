#include<Image.h>
#include<ImageBMP.h>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>

/* .........................................................
*/
void copy_rgb_image(const ImageRGB * p_source,
					ImageRGB * p_destination) {

  // fprintf (stderr, " vamos a copiar imagen ");

  /* free memory of p_destination */
  if ((*p_destination).pixels != 0) {
	free ( (*p_destination).pixels );
	(*p_destination).pixels  = 0;
  }

  /* copy all */
  (*p_destination) = (*p_source); 
  (*p_destination).pixels = 0;

  // get memory for the pixels 
  if ((*p_destination).overall_size > 0) {
	(*p_destination).pixels = (unsigned char *) malloc ( (*p_destination).overall_size ); 
	if ( (*p_destination).pixels == 0 ) {
	  fprintf(stderr, " copy_rgb_image(): malloc() error getting memory\n");
	  throw;
	}

	// copy the bytes
	memcpy 
	  ( (*p_destination).pixels, (*p_source).pixels,  (*p_source).overall_size);
  }

  // fprintf (stderr, " ya he copiado \n");

} /* () */

