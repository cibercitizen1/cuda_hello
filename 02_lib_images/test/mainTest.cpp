//---------------------------------------------------------
// mainTest.cpp
//---------------------------------------------------------

//.........................................................
#include <iostream>
using namespace std;

#include <ImageBMP.h>

//.........................................................
int main_two () {

  cout << "hi" << endl;

  PixelRGB blue = {255, 100, 100};
  ImageBMP image1 (512, 512);

  ImageBMP image2 (512, 512, blue);

  image1.save_to_file("kk.bmp");
  image2.save_to_file("qq.bmp");

  return 0;
} // main ()

//.........................................................
int main() {

  cout << "hola" << endl;

  ImageBMP image1 (0, 0);

  // ImageBMP image2 ("Lena.bmp");
  ImageBMP image2 ("patoC.bmp");
  // ImageBMP image2 ("Mona.bmp");
  // ImageBMP image2 ("prova.bmp");
  // ImageBMP image2 ("prova2.bmp");

  cout << " image 1" << endl;
  image1.print_information();

  cout << endl << " loaded image" << endl;
  image2.print_information();

  cout << " let's copy" << endl;
  image1 = image2;

  cout << " info of the copied image " << endl;
  image1.print_information();

  /*

  image.save_to_file ("kk.bmp");

  Image * pIm = read_image_from_file("kk.bmp");  // polymorphic

  (*pIm).save_image_to_file("qq.bmp");
  */

  cout << "bye bye" << endl;

  image1.save_to_file ("kk.bmp");

} // main ()

//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------
