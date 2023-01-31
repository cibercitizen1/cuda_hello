
/* metricas.h */

#ifndef METRICAS_H
#define METRICAS_H

/* .........................................................
*/
#include <Image.h>

/* .........................................................
*/

#ifdef __CUDA_ARCH__
#define IF_GPU __device__
#else
#define IF_GPU 
#endif

/* .........................................................
*/

IF_GPU float differences_distance (PixelRGB* p1, PixelRGB* p2); 

IF_GPU float euclidean_distance (PixelRGB* p1, PixelRGB* p2);

IF_GPU float fuzzy_distance_1 (PixelRGB* p1, PixelRGB* p2);

IF_GPU void mean_of_pixels(PixelRGB* pix[], unsigned int n, PixelRGB * res); 

IF_GPU void mean_of_pixels_(PixelRGB pix[], unsigned int n, PixelRGB * res); 

/* .........................................................
*/
#endif

