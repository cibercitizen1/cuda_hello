// -*- mode: c++ -*-
// ===================================================================
// mainGeneral.cu
// ===================================================================

#include <stdio.h>
#include <assert.h>
//#include <cuda.h>
// #include <typeinfo>

#include <util.h>
#include <CommandLineParser.h>

#include <cudaUtil.h>

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
// ===================================================================
// As long as  we are using generated data (i.e. not read from a file),
// we choose its size here.
const unsigned int WIDTH_COLUMNS_X = 128; //512;
const unsigned int HEIGHT_ROWS_Y = 256; //512;

// type of the elements on the 2D input data
typedef float Element_Type;

// type of the results
typedef float Result_Type;


// ===================================================================
//
// kernel
//
// ===================================================================
__global__ void test_kernel_1( Result_Type * p_results,
							   unsigned int width,
							   unsigned int height,
							   cudaTextureObject_t in_data_texture
							   ) {

  unsigned int x_column = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int y_row = (blockIdx.y * blockDim.y) + threadIdx.y;

  Element_Type input_val =
	tex2D<Element_Type>( in_data_texture, x_column+0.5f, y_row+0.5f );

  p_results[ (width * y_row) + x_column ] = -input_val;
	
} // ()

// ===================================================================
// ===================================================================
template<typename T>
auto make_up_some_data(
					   unsigned int NUM_ROWS,
					   unsigned int NUM_COLUMNS
					   ) {
  
  //
  // Malloc on the host
  //
  T (*p_data)[NUM_COLUMNS] =
	(T (*)[NUM_COLUMNS]) my_malloc_2D<T>( NUM_ROWS, NUM_COLUMNS );
  // The casting to T (*)[NUM_COLUMNS] is for
  // using 2D indexing (i.e [i][j]) instead
  // of doing the maths [ i*num_cols + j ] ourselves.

  printf( " got the memory for input data \n" );

  //
  // Fill in the data
  // Each element is a float: row.col. Ex. 10.15 is row 10, col 15
  //
  for ( unsigned int row = 0; row < NUM_ROWS; row++ ) {
	//printf( " row %d \n", row );
	for ( unsigned int col = 0; col < NUM_COLUMNS; col++ ) {
	  p_data[ row ][ col ] = row + col/1000.0;
	} // for
  } // for

  //printf( " %f \n", p_data2[ 10*WIDTH_COLUMNS_X + 15 ] );
  printf( " %f \n", p_data[10][15] );

  
  //
  //
  //
  //return p_data;
  return (T (*)[]) p_data;

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
  // Create the input data for the kernels
  // to compute something on it
  // .................................................................
  auto p_data =
	make_up_some_data<Element_Type>( HEIGHT_ROWS_Y, WIDTH_COLUMNS_X );
  
  printf( " input data generated \n" );
 
  // .................................................................
  // Copy the input data to the device, in a texture
  // .................................................................
  Texture_Memory_Holder<Element_Type>
	data_in_texture( p_data, HEIGHT_ROWS_Y, WIDTH_COLUMNS_X ); 

  printf( " placed input data in device memory, bound to a texture \n" );
					   
  // .................................................................
  // Get memory to hold the results (on the GPU and on the CPU)
  // Let's suppose that we get a result for each input element.
  // .................................................................
  Results_Holder<Result_Type>
	results( HEIGHT_ROWS_Y, WIDTH_COLUMNS_X );

  printf( " got data for the results \n" );
				   
  // .................................................................
  // set up the launch of kernels
  // .................................................................
  // Number of blocks we need considering a thread per element (pixel)
  // in the 2D data
  // Defined in 2D.
  //
  dim3 NUM_BLOCKS( WIDTH_COLUMNS_X / THREADS_PER_BLOCK.x,
				   HEIGHT_ROWS_Y / THREADS_PER_BLOCK.y );

  // .................................................................
  // Launch the kernel
  // .................................................................
  printf( " launching kernels \n" );
  test_kernel_1<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>
	(
	 results.results_on_device,
	 WIDTH_COLUMNS_X,
	 HEIGHT_ROWS_Y,
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

  // show sth. to check if the kernel has done something
  printf( " %f \n", results(10, 15) );
  printf( " %f \n", results(25, 40) );
  printf( " %f \n", results[5][5] );

  //printf( " %f \n", results.results_on_host[10][15] );

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

  pause( " the end " );
} // ()

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
