//---------------------------------------------------------
// mainTest.cpp
//---------------------------------------------------------

//.........................................................
#include <iostream>
using namespace std;

#include <ImagenBMP.h>

//.........................................................
int mainOtro () {

  cout << "hola" << endl;

  PixelRGB azul = {255, 100, 100};
  ImagenBMP imagen1 (512, 512);

  ImagenBMP imagen2 (512, 512, azul);

  imagen1.guardarEnFichero ("kk.bmp");
  imagen2.guardarEnFichero ("qq.bmp");

  return 0;
} // main ()

//.........................................................
int main() {

  cout << "hola" << endl;

  ImagenBMP imagen1 (0, 0);

  // ImagenBMP imagen2 ("Lena.bmp");
  ImagenBMP imagen2 ("patoC.bmp");
  // ImagenBMP imagen2 ("Mona.bmp");
  // ImagenBMP imagen2 ("prova.bmp");
  // ImagenBMP imagen2 ("prova2.bmp");

  cout << " la mia " << endl;
  imagen1.muestraInfo();

  cout << endl << " una correcta " << endl;
  imagen2.muestraInfo();

  cout << " vamos a copiar " << endl;
  imagen1 = imagen2;

  cout << " la mia copiada de la correcta" << endl;
  imagen1.muestraInfo();

  /*

  imagen.guardarEnFichero ("kk.bmp");

  Imagen * pIm = leerImagenDeFichero("kk.bmp");  // polimorfico

  (*pIm).guardarEnFichero("qq.bmp");
  */

  cout << "adios" << endl;

  imagen1.guardarEnFichero ("kk.bmp");

} // main ()
