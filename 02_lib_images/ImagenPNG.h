/* ImagenPNG.h */
/* NO UTILIZAR */

#ifndef IMAGENPNG_H
#define IMAGENPNG_H

#include<Imagen.h>

/* .........................................................
*/
#pragma pack(2) /*2 byte packing */ 
typedef struct { 
  unsigned short int type; 
  unsigned int size; 
  unsigned short int reserved1,reserved2; 
  unsigned int offset; 
} PNGCambiarFichCabecera1;

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
} PNGCambiarFichCabecera2;

/* .........................................................
*/
class ImagenPNG : public Imagen 
{
 private:
  PNGCambiarFichCabecera1 cabecera1;
  PNGCambiarFichCabecera2 cabecera2;
  void destruir();
 public:
  ImagenRGB  laImagen;
  ImagenPNG(const char * nombreFich) ;
  void guardarEnFichero(const char * nombreFich);
  ~ImagenPNG();
}; /* class */

#endif 

