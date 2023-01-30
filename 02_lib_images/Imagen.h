/* Imagen.h */

#ifndef IMAGEN_H
#define IMAGEN_H

/* .........................................................
*/
typedef struct ImagenRGB {
 public:
  unsigned int ancho;
  unsigned int alto;
  unsigned int bitsPixel;  /* 1 2 8 24(RGB)  32(RGB+alpha) */
  unsigned char * pixels; /* array de  bytes */
  unsigned int bytesFila;
  unsigned int tamanyoBruto;  // 4
} ImagenRGB;

/* para acceder a un pixel dado un puntero al anterior
 * struct y (f,c) */
#define elPixel(imPtr, f,c) \
  (PixelRGB*) & (*(imPtr)).pixels[((f)*(*imPtr).bytesFila)+(3*(c))]

void copiarImagenRGB(const ImagenRGB * origen, ImagenRGB * destino);

/* .........................................................
*/
typedef struct {
  unsigned int fila;
  unsigned int columna;
} Coordenada;

/* .........................................................
*/
typedef struct {
  unsigned char b;
  unsigned char g; /* atencion lo dejo en BGR */
  unsigned char r;
} PixelRGB;

/* .........................................................
*/
typedef struct {
  unsigned char b;
  unsigned char g; /* atencion lo dejo en BGR */
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
class ErrorFicheroNoEncontrado { };
class ErrorDeLecturaDeFichero { };

/* .........................................................
*/
class Imagen {
 public:
  ImagenRGB * laImagen;
  virtual void guardarEnFichero(const char * nombreFich) const = 0;
  virtual ~Imagen() { }
}; /* class */

#endif 

