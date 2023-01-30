/* ImagenBMP.h */

#ifndef IMAGENBMP_H
#define IMAGENBMP_H

#include<Imagen.h>

/* .........................................................
*/
#pragma pack(2) /*2 byte packing */ 
typedef struct { 
  unsigned short int type; 
  unsigned int size; 
  unsigned short int reserved1,reserved2; 
  unsigned int offset; 
} BmpFichCabecera1;

/* .........................................................
*/
#pragma pack() /* Default packing */ 
typedef struct { 
  unsigned int size;  // 4
  int width; // 4
  int height;  // 4
  unsigned short int planes;  // 2
  unsigned short int bitsPixel;  // 2
  unsigned int compression;  // 4
  unsigned int imagesize;  // 4
  int xresolution,yresolution; 
  unsigned int ncolors;  // 4
  unsigned int importantcolors;  // 4
} BmpFichCabecera2;

/* .........................................................
*/

/* .........................................................
*/
class ImagenBMP : public Imagen 
{
 private:
  BmpFichCabecera1 cabecera1;
  BmpFichCabecera2 cabecera2;
  void liberarPixels();
  static const PixelRGB pixelBlanco; /* constante pero se incializa en el .cpp */

 public:

  ImagenRGB  laImagen; /* ojo: idealmente deberia ser privada,
						* pero es publica para que se pueda
						* acceder "a bajo nivel" desde 
						* programas "solo en C" */

  ImagenBMP(const char * nombreFich) ;

  ImagenBMP(const unsigned int alto, 
			const unsigned int ancho,
			PixelRGB color = pixelBlanco);

  /* acceso a pixel dado f, c */
  PixelRGB * pixel (const unsigned int f, const unsigned int c);

  void guardarEnFichero(const char * nombreFich) const;

  void muestraInfo() const;

  void operator=(const ImagenBMP &);

  ~ImagenBMP();
}; /* class */

/* .........................................................
*/
void copiaPoniendoPading3a4 (const unsigned char * fuente, 
						unsigned char * destino,
						unsigned int tamanyo);

void copiaQuitandoPading4a3 (const unsigned char * fuente, 
						unsigned char * destino,
						unsigned int tamanyo);

#endif 

