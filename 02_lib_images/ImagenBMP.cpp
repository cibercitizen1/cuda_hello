/* ImagenBMP.cpp */

#include <ImagenBMP.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* .........................................................
*/
ImagenBMP::ImagenBMP(const char *  nombreFich) {
  FILE *fich;

  /* 
   * 
   */
  laImagen.pixels = 0;

  /* abro el fichero */
  fich = fopen(nombreFich,"r"); 
  if (fich == 0) {
	liberarPixels();
	throw new ErrorFicheroNoEncontrado();
  }

  /* leo las cabeceras */
  fread( & cabecera1,sizeof(BmpFichCabecera1),1,fich); 
  fread( & cabecera2,sizeof(BmpFichCabecera2),1,fich); 

  /* averiguo ancho, alto y bits x pixel */
  laImagen.ancho = cabecera2.width;
  laImagen.alto = cabecera2.height;

  laImagen.tamanyoBruto = cabecera2.imagesize;
  
  laImagen.bytesFila = laImagen.tamanyoBruto / laImagen.alto;

  laImagen.bitsPixel = cabecera2.bitsPixel; 

  /* leo los pixels */
  /* posicionamos en el fichero para leer los pixels */

  if ( fseek (fich, cabecera1.offset,SEEK_SET) != 0 ) {
	/* printf (" no nos hemos posicionado para leer \n"); */
	liberarPixels();
	throw new ErrorDeLecturaDeFichero();
  }

  unsigned long int bytesQueLeer = laImagen.tamanyoBruto;

  /* crear la memoria para el array de bytes */
  laImagen.pixels = (unsigned char *) malloc ( bytesQueLeer ); 
  if ( laImagen.pixels == 0 ) {
	liberarPixels();
	printf( " error en malloc() para guardar imagen en memoria \n");
	throw;
  }

  /* leerlos del fichero */
  unsigned long int leidos;
  leidos = fread(laImagen.pixels, 1, bytesQueLeer, fich); 
  if ( leidos != bytesQueLeer ) {
	liberarPixels();
	throw new ErrorDeLecturaDeFichero();
  }

  /* cierro el fichero */
  fclose (fich);
}


/* .........................................................
*/
const PixelRGB ImagenBMP::pixelBlanco = {255,255,255};

ImagenBMP::ImagenBMP(const unsigned int alto, const unsigned int ancho,
					 PixelRGB color) {

  // relleno los campos de ImagenRGB
  laImagen.alto = alto;
  laImagen.ancho = ancho;
  laImagen.bitsPixel = 24; // 24 bits = 8 bytes <= RGB
  laImagen.bytesFila = (ancho * 3);
  laImagen.bytesFila += (4-(laImagen.bytesFila%4))%4; // anyadir padding
  laImagen.pixels = 0;
  laImagen.tamanyoBruto = laImagen.bytesFila * laImagen.alto;

  // memoria para los pixels de la  imagen
  if (laImagen.tamanyoBruto > 0) {
	laImagen.pixels = (unsigned char *) malloc ( laImagen.tamanyoBruto ); 
	if ( laImagen.pixels == 0 ) {
	  liberarPixels();
	  printf( " error en malloc() para guardar imagen en memoria \n");
	  throw;
	}
	// pongo los pixels  al color indicado
	// memset (laImagen.pixels, 255, laImagen.tamanyoBruto); antes siempre a blanco
	PixelRGB * pAux = (PixelRGB*) laImagen.pixels;
	for (int i = 0; i<= (laImagen.alto*laImagen.ancho)-1; i++) {
	  pAux[i] = color;
	}
  }

  // relleno las cabeceras del fichero .bmp

  /* cabecera 1 */
  cabecera1.type = 19778;   /* espero que funcione debe ser los
							   caracteres BM? del principio del fichero */
  cabecera1.offset = 54;  /* tamanyo en bytes de las 2 cabeceras
						   * se ve que siempre es 54 */
  cabecera1.size = laImagen.tamanyoBruto + cabecera1.offset;
  /* los campos reserved creo que no se usan, los pongo a 0 */
  cabecera1.reserved1 = 0;
  cabecera1.reserved2 = 0;

  /* cabecera 2 */
  cabecera2.size = 40; /* tamanyo de esta cabecera en bytes, creo */
  cabecera2.width = laImagen.ancho;
  cabecera2.height = laImagen.alto; 
  cabecera2.planes = 1;
  cabecera2.bitsPixel = laImagen.bitsPixel;
  cabecera2.compression = 0;
  cabecera2.imagesize = laImagen.tamanyoBruto; 
  cabecera2.xresolution = 2835;
  cabecera2.yresolution = 2835;
  cabecera2.ncolors = 0;
  cabecera2.importantcolors = 0;

} 

/* .........................................................
*/
void ImagenBMP::liberarPixels() {
  if (laImagen.pixels == 0) {
	return;
  }
  free ( laImagen.pixels );
  laImagen.pixels  = 0;
}

/* .........................................................
*/
ImagenBMP::~ImagenBMP() {
  liberarPixels();
}

/* .........................................................
*/
void ImagenBMP::guardarEnFichero(const char * nombreFich) const {
  FILE * fout;
  fout = fopen(nombreFich,"w"); 

  if (fout == 0) {
	return;
  }

  fwrite( & cabecera1, sizeof(BmpFichCabecera1),1,fout); 
  fwrite( & cabecera2, sizeof(BmpFichCabecera2),1,fout); 
  fwrite( laImagen.pixels, 1 , cabecera2.imagesize, fout);

  fclose (fout);
}

/* .........................................................
*/
void ImagenBMP::muestraInfo() const {
  printf (" ImagenRGB ---------\n ");
  printf (" alto=%d ancho=%d bytesFila=%d bytesFilaSinPad=%d tamanyoBruto=%d\n",
		  laImagen.alto,
		  laImagen.ancho,
		  laImagen.bytesFila,
		  laImagen.ancho*3,
		  laImagen.tamanyoBruto
		  );

  printf (" cabecera cabecera 1 ---------\n ");
  printf (" type=%d, size=%d, offset=%d\n", 
		  cabecera1.type,
		  cabecera1.size,
		  cabecera1.offset);
  printf (" cabecera cabecera 2 ---------\n ");
  printf (" size=%d, width=%d, heigth=%d, planes=%d, bitsPixel=%d, compression=%d, imagesize=%d, xresolution=%d, yresolution=%d, ncolors=%d, importantcolors=%d\n", 
		  cabecera2.size,
		  cabecera2.width,
		  cabecera2.height,
		  cabecera2.planes,
		  cabecera2.bitsPixel,
		  cabecera2.compression,
		  cabecera2.imagesize,
		  cabecera2.xresolution,
		  cabecera2.yresolution,
		  cabecera2.ncolors,
		  cabecera2.importantcolors);
}

/* .........................................................
*/
void ImagenBMP::operator=(const ImagenBMP & otra) {


  liberarPixels();

  /* copio todo las cabeceras*/

  cabecera1 = otra.cabecera1;
  cabecera2 = otra.cabecera2;

  /* copio la imagenRGB incluyendo los pixels */
  copiarImagenRGB(& otra.laImagen,  &laImagen);

}

/* .........................................................
*/
PixelRGB * ImagenBMP::pixel (const unsigned int f, const unsigned int c) {
  return elPixel ( &(*this).laImagen, f, c);
} // ()

/* .........................................................
*/
void copiaPoniendoPading3a4 (const unsigned char * fuente, 
							 unsigned char * destino,
							 unsigned int tamanyoFuente) 
{
  unsigned int j=0;
  for (unsigned int i=0; i<=tamanyoFuente-1; i++) 
	{
	  destino[j] = fuente[i];
	  if (i%3 == 2) {
		j++;
		destino[j] = 0;
	  }

	  j++;
	}
} // ()

/* .........................................................
*/
void copiaQuitandoPading4a3 (const unsigned char * fuente, 
						unsigned char * destino,
						unsigned int tamanyoFuente)
{
  
  unsigned int j=0;
  for (unsigned int i=0; i<=tamanyoFuente-1; i++) 
	{
	  if ( (i%4) < 3 ) 
		{
		  destino[j] = fuente[i];
		  j++;
		}
	}
} // ()
