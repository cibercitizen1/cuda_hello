
/* metricas.h */

#ifndef METRICAS_H
#define METRICAS_H

/* .........................................................
*/
#include <Imagen.h>

/* .........................................................
*/

#ifdef USAR_CUDA
#define SI_EN_GPU __device__
#else
#define SI_EN_GPU 
#endif

/* .........................................................
*/

SI_EN_GPU float distanciaDiferencias (PixelRGB* p1, PixelRGB* p2); 
SI_EN_GPU float distanciaEuclidea (PixelRGB* p1, PixelRGB* p2); 
SI_EN_GPU float distanciaFuzzy1 (PixelRGB* p1, PixelRGB* p2); 

SI_EN_GPU void media(PixelRGB* pix[], unsigned int n, PixelRGB * res); 

SI_EN_GPU void mediaC(PixelRGB pix[], unsigned int n, PixelRGB * res); 

/* .........................................................
*/
#endif

