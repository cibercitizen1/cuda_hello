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
#include <cudaImageUtil.h>
#include <metrics.h>

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
//const unsigned int BLOCK_SIDE = 32;
const unsigned int BLOCK_SIDE = 2;
dim3 THREADS_PER_BLOCK( BLOCK_SIDE, BLOCK_SIDE );

// ===================================================================
// ===================================================================
// Please,read cudaImageUtil.h
	
#include <metrics.cpp> // it seems that __device__ functions
// must be in the same compilation unit

#include <kernel_detect.cu>

#define WHAT_KERNEL_TO_RUN kernel_detect

// ===================================================================
//
// kernel
//
// ===================================================================
__global__ void test_kernel_1( cuda4bytes * p_results, // -> results
							   const unsigned int width, // <-
							   const unsigned int height, // <- 
							   cudaTextureObject_t in_data_texture // <- data in
							   ) {

  // find out this thread's coordinates
  const unsigned int x_column = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int y_row = (blockIdx.y * blockDim.y) + threadIdx.y;

  // read the pixel assigned to this thread
  PixelRGBA input_pixel = access_pixel( in_data_texture, x_column, y_row );
  
  //
  // do some processing
  // delete the blue value
  //
  input_pixel.b = 128;
  
  //
  // copy the new value to results
  //
  p_results[ (width * y_row) + x_column ] = my_rgba_to_cuda4bytes( input_pixel );

  //
  // control setting
  //
  if (x_column == y_row || x_column == width-1 ) {
	cuda4bytes uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }
	
  //PixelRGBA new_pixel { .b=200, .g=123, .r=123, .a=123 };
  //cuda4bytes new_cuda4bytes = make_cuda4bytes( 200, 10, 20, 1 );
} // ()

// ===================================================================
//
// main
//
// ===================================================================
int main( int n_args, char * args[] ) {

  unsigned int t0 = 0;
  unsigned int t1 = 0;

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
  printf( " loading image %s\n", options.option_value<string>("image").c_str() );
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

  // .................................................................
  // The image already has in memory its pixels.
  // But because bmp are 3 bytes per pixel,
  // we are now allocating host memory to copy the original
  // pixels adding a padding 4th byte.
  // .................................................................
  cuda4bytes* input_4byte_pixels = 
	get_memory_and_copy_with_padding_3to4( & an_image );
  
  // .................................................................
  // start the clock
  // .................................................................
  t0 = clock();

  // .................................................................
  // Copy the input data to the device, in a texture
  // .................................................................
  Texture_Memory_Holder<cuda4bytes>
	data_in_texture(  input_4byte_pixels, // from
					  an_image.the_image.height, // h x w
					  an_image.the_image.width );

  printf( " placed input data in device memory, bound to a texture \n" );
					   
  // .................................................................
  // Get memory to hold the results (on the GPU and on the CPU)
  // Let's suppose that we get a result for each input element.
  // .................................................................
  Results_Holder<cuda4bytes>
	results(an_image.the_image.height, an_image.the_image.width );
  //results(an_image.the_image.height, an_image.the_image.width*4 ); AQUI

  printf( " got data for the results \n" );

  // .................................................................
  // Launch the kernel
  // .................................................................
  printf( " launching kernels \n" );
	//test_kernel_2<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>

  WHAT_KERNEL_TO_RUN<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>
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
  // stop the clock
  // .................................................................
  t1 = clock();

  // .................................................................
  // from input_4byte_pixels to pixels, removing padding ( fourth byte )
  // now copy froom results.results_on_host
  // to the pointer pixels in the image, so we can save them
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
  cudaFree( input_4byte_pixels );
  
  // The memory for the results ( host and device ) is freed by
  // the destructor of results

  // The memory on the texture is freed by
  // the destructor of data_texture

  // .................................................................
  // .................................................................
  printf( " ********************************************* \n" );
  printf( " elapsed time = %d \n", t1-t0 );
  printf( " ********************************************* \n" );

  // .................................................................
  // .................................................................
  printf( " all done \n" );

  //pause( " the end " );
} // ()

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
