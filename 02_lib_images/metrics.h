
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

IF_GPU float differences_distance (PixelRGBA* p1, PixelRGBA* p2); 

IF_GPU float euclidean_distance (PixelRGBA* p1, PixelRGBA* p2);

IF_GPU float fuzzy_distance_1 (PixelRGBA* p1, PixelRGBA* p2);

IF_GPU void mean_of_pixels(PixelRGBA pix[], unsigned int n, PixelRGBA * res); 

IF_GPU void mean_of_pixels_wtf(PixelRGBA* pix[], unsigned int n, PixelRGBA * res); 

/* .........................................................
*/
#endif

