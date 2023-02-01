// -*- mode: c++ -*-
// ===================================================================
// mainGeneral.cu
// ===================================================================

#include <stdio.h>
#include <assert.h>

#include <util.h>
#include <CommandLineParser.h>
#include <ImageBMP.h>

#include <cudaUtil.h>

//#include <cuda.h>
// #include <typeinfo>


// ===================================================================
// ===================================================================

// The maximum number of threads in a block is 1024 for some many
// modern (2022) cuda devices. Hence, 32x32 is the max. dimensions
// for a bloock

// But that number of threads per block is influenced by how much
// local memory a kernel uses. The more memory used, the lesser
// number of threads a block can have.
//
// We define the block's number of threads in 2D for we will be
// operate on 2D data.

//const unsigned int BLOCK_SIDE = 8;
const unsigned int BLOCK_SIDE = 32;
dim3 THREADS_PER_BLOCK( BLOCK_SIDE, BLOCK_SIDE );

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
// A "4 bytes struct" is defined in cuda as uchar4.
// As a consequence, we will be using uchar4.
//
// Addition predefined cuda functions
//                                 blue, green red, alpha?
// uchar4 new_pixel = make_uchar4( 200, 10, 20, 1 );
//
//
// Let's define some helper functions for us.

//
//
//
__device__ inline PixelRGBA my_uchar4_to_rgba( uchar4 src ) {
  PixelRGBA dest;

  dest.r = src.z;
  dest.g = src.y;
  dest.b = src.x;
  dest.a = src.w;

  return dest;
} // ()

//
//
//
__device__ inline uchar4 my_rgba_to_uchar4( PixelRGBA src ) {
  uchar4 dest;
  
  dest.z = src.r;
  dest.y = src.g;
  dest.x = src.b;
  dest.w = src.a;

  return dest;
} // ()

//
//
//
template<typename Type = uchar4>
__device__ inline auto access_element( cudaTextureObject_t tex,
									   unsigned int x_column,
									   unsigned int y_row ) {
  return tex2D<Type>( tex, x_column+0.5f, y_row+0.5f );
} // ()
	
// ===================================================================
//
// kernel
//
// ===================================================================
__global__ void test_kernel_2( uchar4 * p_results,
							   unsigned int width,
							   unsigned int height,
							   cudaTextureObject_t in_data_texture
							   ) {

  // find out this thread's coordinates
  unsigned int x_column = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int y_row = (blockIdx.y * blockDim.y) + threadIdx.y;

  // read the pixel assigned to this thread
  uchar4 input_uchar4 = access_element( in_data_texture, x_column, y_row );
  PixelRGBA input_pixel = my_uchar4_to_rgba( input_uchar4 );
  
  // delete the blue value
  input_pixel.b = 0;
  
  //
  uchar4 new_uchar4 = my_rgba_to_uchar4( input_pixel );
	
  p_results[ (width * y_row) + x_column ] = new_uchar4;

  if (x_column == y_row || x_column == width-1 ) {
	uchar4 uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }
	
} // ()

// ===================================================================
//
// kernel
//
// ===================================================================
__global__ void test_kernel_1( uchar4 * p_results,
							   unsigned int width,
							   unsigned int height,
							   cudaTextureObject_t in_data_texture
							   ) {

  // find out this thread's coordinates
  unsigned int x_column = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int y_row = (blockIdx.y * blockDim.y) + threadIdx.y;

  // read the pixel assigned to this thread
  //uchar4 input_pixel = tex2D<uchar4>( in_data_texture, x_column+0.5f, y_row+0.5f );
  uchar4 input_pixel = access_element( in_data_texture, x_column, y_row );
  
  // make up a new pixel
  PixelRGBA new_pixel { .b=200, .g=123, .r=123, .a=123 };
  
  //
  uchar4 new_uchar4 = my_rgba_to_uchar4( new_pixel );
  
  //uchar4 new_uchar4 = make_uchar4( 200, 10, 20, 1 );
	
  p_results[ (width * y_row) + x_column ] = new_uchar4;

  if (x_column == y_row || x_column == width-1 ) {
	uchar4 uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }
	
} // ()

// ===================================================================
//
// main
//
// ===================================================================
int main( int n_args, char * args[] ) {

  printf( " starting \n" );
  
  Command_Line_Parser options( n_args, args );

  // .................................................................
  // pick the fastest device
  // .................................................................
  cudaSetDevice( cudaGetMaxGflopsDeviceId() );

  // .................................................................
  // load a bmp image
  // .................................................................
  printf( "------------------------------------------------------\n" );
  printf( " loaded image %s\n", options.option_value<string>("image").c_str() );
  printf( "------------------------------------------------------\n" );
  
  ImageBMP an_image( options.option_value<string>("image").c_str() );

  an_image.print_information();
  printf( "------------------------------------------------------\n" );
  printf( "------------------------------------------------------\n" );

  // .................................................................
  // set up the number of kernel blocks accordingly to
  // the image dimension
  // .................................................................
  // Number of blocks we need considering a thread per element (pixel)
  // in the 2D data
  // Defined in 2D.
  //
  dim3 NUM_BLOCKS( an_image.the_image.width / THREADS_PER_BLOCK.x,
				   an_image.the_image.height / THREADS_PER_BLOCK.y );

  printf( " num blocs x =  %d \n", NUM_BLOCKS.x );
  printf( " threads in x %d \n", THREADS_PER_BLOCK.x );
  printf( " num blocs y =  %d \n", NUM_BLOCKS.y );
  printf( " threads in y %d \n", THREADS_PER_BLOCK.y );
  printf( " total threads ---> %d \n", NUM_BLOCKS.x * THREADS_PER_BLOCK.x * NUM_BLOCKS.y * THREADS_PER_BLOCK.y );

  printf( "------------------------------------------------------\n" );
  printf( "------------------------------------------------------\n" );
  uchar4* p_data = // matrix of 4 unsigned char
	my_malloc<uchar4>( an_image.the_image.height,
					   an_image.the_image.width); 

  // the bmp image has 3 unsigned bytes per pixel
  // (there are actually .overall_size bytes)
  // So, we add a fourth.
  copy_adding_padding_3to4( an_image.the_image.pixels, // from
							(unsigned char *) p_data, // to
							an_image.the_image.overall_size );
 
  // .................................................................
  // Copy the input data to the device, in a texture
  // .................................................................
  printf( " %d == %d ? \n", an_image.the_image.width*4, 
		  an_image.the_image.bytes_per_row );

  Texture_Memory_Holder<uchar4>
	data_in_texture(  p_data,
					  an_image.the_image.height,
					  an_image.the_image.width );
  //an_image.the_image.width * 4);

  printf( " placed input data in device memory, bound to a texture \n" );
					   
  // .................................................................
  // Get memory to hold the results (on the GPU and on the CPU)
  // Let's suppose that we get a result for each input element.
  // .................................................................
  Results_Holder<uchar4>
	results(an_image.the_image.height, an_image.the_image.width );
  //results(an_image.the_image.height, an_image.the_image.width*4 ); AQUI

  printf( " got data for the results \n" );

  // .................................................................
  // Launch the kernel
  // .................................................................
  printf( " launching kernels \n" );
  test_kernel_2<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>
	(
	 results.results_on_device,
	 an_image.the_image.width,
	 an_image.the_image.height,
	 data_in_texture.texture
	 );
  

  // .................................................................
  // wait
  // .................................................................
  cudaDeviceSynchronize();

  check_cuda_call( " kernels done\n" );
  
  printf( " kernels done \n" );

  // .................................................................
  // Copy results from memory device
  // .................................................................
  results.copy_results_device_to_host();

  // .................................................................
  // from p_data to pixels, removing padding ( fourth byte )
  // .................................................................
  copy_removing_padding_4to3((unsigned char *) results.results_on_host, // from
							 an_image.the_image.pixels, // to
							 results.size );
  //an_image.the_image.height *  an_image.the_image.width * 4 );
  // .................................................................
  // save to disk
  // .................................................................
  an_image.save_to_file( "corrected.bmp" );

  // .................................................................
  // free the memory
  // .................................................................
  // Memory on the host (CPU)
  cudaFree( p_data );
  
  // The memory for the results ( host and device ) is freed by
  // the destructor of results

  // The memory on the texture is freed by
  // the destructor of data_texture

  // .................................................................
  // .................................................................
  printf( " all done \n" );

  //pause( " the end " );
} // ()

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
