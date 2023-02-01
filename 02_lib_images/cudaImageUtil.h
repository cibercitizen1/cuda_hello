// -*- mode: c++ -*-
// ===================================================================
// cudaImageUtil.h
// ===================================================================
#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

//#include <cudaUtil.h>

// ===================================================================
// ===================================================================
//
// A byte (i.e. 8 bits) is an unsigned char. We treat arrays of bytes
// as a arrays of unsigned bytes at low level.
//
// Despite that, later, the elements of the array can be accessed 
// in a more appropiate or convenient manner. For instance, if
// if the data are pixels of a BMP image.
//
// A pixel of a BMP image has 3 bytes. When copied to host and device
// memory, a 4th byte is added in order to have the accesses aligned
// to a even number. This speeds up the accesses.
//
// We defined a PixelRGBA struct (see Image.h) supposedly having
// 4 bytes, but it appears no to work well with cuda functions.
//
// A "4 bytes struct" is defined in cuda as cuda4bytes.
// As a consequence, we will be using cuda4bytes.
using cuda4bytes = uchar4;
//
// Addition predefined cuda functions
//                                 blue, green red, alpha?
// cuda4bytes new_pixel = make_cuda4bytes( 200, 10, 20, 1 );
//
//
// Let's define some helper functions for us.

// ...................................................................
// ...................................................................
__device__ inline PixelRGBA my_cuda4bytes_to_rgba( cuda4bytes & src ) {
  PixelRGBA dest;

  dest.r = src.z;
  dest.g = src.y;
  dest.b = src.x;
  dest.a = src.w;

  return dest;
} // ()

// ...................................................................
// ...................................................................
__device__ inline cuda4bytes my_rgba_to_cuda4bytes( PixelRGBA & src ) {
  cuda4bytes dest;
  
  dest.z = src.r;
  dest.y = src.g;
  dest.x = src.b;
  dest.w = src.a;

  return dest;
} // ()

// ...................................................................
// ...................................................................
template<typename Type = cuda4bytes>
__device__ inline auto access_element( const cudaTextureObject_t & tex,
									   unsigned int x_column,
									   unsigned int y_row ) {
  return tex2D<Type>( tex, x_column+0.5f, y_row+0.5f );
} // ()


// ...................................................................
// ...................................................................
__device__ inline auto access_pixel( const cudaTextureObject_t & tex,
									 unsigned int x_column,
									 unsigned int y_row ) {
  cuda4bytes input_cuda4bytes = access_element( tex, x_column, y_row );
  PixelRGBA input_pixel = my_cuda4bytes_to_rgba( input_cuda4bytes );
  return input_pixel;
} // ()

// ===================================================================
// ===================================================================

  // .................................................................
  // An image already has in memory its pixels.
  // But because bmp are 3 bytes per pixel,
  // we are now allocating host memory to copy the original
  // pixels adding a padding 4th byte.
  // .................................................................
inline
cuda4bytes * get_memory_and_copy_with_padding_3to4( Image * p_image ) {

  cuda4bytes* input_4byte_pixels = // matrix of 4 unsigned char
	my_malloc<cuda4bytes>( (*p_image).the_image.height, (*p_image).the_image.width); 

  // the bmp image has 3 unsigned bytes per pixel
  // (there are actually .overall_size bytes)
  // So, we add a fourth.
  copy_adding_padding_3to4( (*p_image).the_image.pixels, // from
							(unsigned char *) input_4byte_pixels, // to
							(*p_image).the_image.overall_size );
 
  return input_4byte_pixels;
} // ()

#endif

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
