#include<Imagen.h>
#include<ImagenBMP.h>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>

/* .........................................................
*/
Imagen * leerImagenDeFichero(const char *  nombreFich) {
  return new ImagenBMP(nombreFich);
}

/* .........................................................
*/
void copiarImagenRGB(const ImagenRGB * origen, ImagenRGB * destino) {

  // fprintf (stderr, " vamos a copiar imagen ");

  /* liberar memoria de los pixels destino */
  if ((*destino).pixels != 0) {
	free ( (*destino).pixels );
	(*destino).pixels  = 0;
  }

  // fprintf (stderr, " memoria liberada ");

  /* copio todo */
  (*destino) = (*origen); 
  (*destino).pixels = 0;

  // fprintf (stderr, " copiados campos ");

  // conseguir memoria para los pixels de la  imagen
  if ((*destino).tamanyoBruto > 0) {
	(*destino).pixels = (unsigned char *) malloc ( (*destino).tamanyoBruto ); 
	if ( (*destino).pixels == 0 ) {
	  fprintf(stderr, " error en malloc() para guardar imagen en memoria \n");
	  throw;
	}

	// copio los bytes
	memcpy 
	  ( (*destino).pixels, (*origen).pixels,  (*origen).tamanyoBruto);
  }

  // fprintf (stderr, " ya he copiado \n");

} /* () */

