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

// The number of threads per block is influenced by how many
// local memory a kernel uses. The more memory used, the lesser
// number of threads a block can have.
//
// We define the block's number of threads in 2D for we will be
// operate on 2D data.

const unsigned int BLOCK_SIDE = 8;
dim3 THREADS_PER_BLOCK( BLOCK_SIDE, BLOCK_SIDE );

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

  unsigned int x_column = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int y_row = (blockIdx.y * blockDim.y) + threadIdx.y;


  //uchar4 input_pixel = tex2D<uchar4>( in_data_texture, x_column+0.5f, y_row+0.5f );
  uchar4 input_pixel = tex2D<uchar4>( in_data_texture, x_column+0.5f, y_row+0.5f );
  
  uchar4 new_pixel = make_uchar4( 10, 10, 200, 1 );
	
  p_results[ (width * y_row) + x_column ] = new_pixel;

  if (x_column == y_row) {
	uchar4 uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }

  /*

  p_results[ (width * y_row) + x_column ] = -input_val;
  */
	
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
  test_kernel_1<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>
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
  // from p_data to pixels, removing padding
  // .................................................................

  copy_removing_padding_4to3((unsigned char *) results.results_on_host, // from
							 an_image.the_image.pixels, // to
							 an_image.the_image.height *  an_image.the_image.width * 4 );
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
