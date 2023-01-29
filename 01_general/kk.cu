// -*- mode: c++ -*-
// ===================================================================
// mainGeneral.cu
// ===================================================================

#include <stdio.h>
#include <assert.h>
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
// ===================================================================
void check_cuda_call( const char * msg = "" ) {
  
  cudaError_t error = cudaGetLastError();
  
  if ( error != cudaSuccess ) {
	  
	printf( " check_cuda_call: failed %s, reason: %s \n", msg, cudaGetErrorString( error ));
	exit(0);
	assert( error == cudaSuccess );
  } 
} // ()

// ===================================================================
// utility for malloc()
// ===================================================================
//class Malloc_Error {};

// ===================================================================
template<typename T>
T my_malloc( const long unsigned size )
// spec. no longer needed: throw ( Malloc_Error )
{
  void * ptr = malloc( size );

  assert( ptr != nullptr && "my_malloc failed" );

  return static_cast<T>( ptr );
} // ()

// ===================================================================
/*
  template<typename T, unsigned int NUM_ROWS, unsigned int NUM_COLUMNS>
  auto my_malloc_2D_OK( ) {
  auto ptr = new T[NUM_ROWS][NUM_COLUMNS];
  if ( ptr == nullptr ) {
  throw Malloc_Error {};
  }
  return ptr;
  } // ()
*/

// ===================================================================
// Let's use cudaMallocHost()
// which gets "pinned memory" in the CPU for us.
// I guess that means that the memory is aligned so that transfers
// from and to the GPU are faster.
template<typename T>
T * my_malloc_2D( unsigned int NUM_ROWS, unsigned int NUM_COLUMNS) {
  
  //
  // compute the size required
  //
  size_t size = NUM_ROWS * NUM_COLUMNS * sizeof( T );

  //printf( "my_malloc_2D(): rows=%d columns=%d, size=%d\n", NUM_ROWS, NUM_COLUMNS, size );

  //
  // malloc
  //
  T* ptr = nullptr;
  cudaMallocHost( & ptr, size );

  check_cuda_call( "my_malloc_2D(): cudaMallocHost()" );
  
  //
  // make sure we've got memory
  //
  assert( ptr != nullptr && "my_malloc_2D failed" );

  //return ( T * [] ) ptr;
  return ptr;
} // ()

/*
//auto kk = new int [10][20];
// OK int (* kk)[20] = new int [10][20];
int (* kk)[20] = new int [10][20];
kk[9][2] = 13;
*/

// ===================================================================
// Utility class for allocating memory both on the device
// and on the host.
// ===================================================================
template<typename T>
class Results_Holder {
private:
  const unsigned int NUM_ROWS;
  const unsigned int NUM_COLUMNS;
public:
  T * results_on_host;
  T * results_on_device;

  // -----------------------------------------------------------------
  // Used to access to the correct row of results_on_host.
  // Column dimension is required to get the correct one.
  // Because a pointer is returned, [] can be chained:
  // Example:
  // results[10][15]
  // -----------------------------------------------------------------
  //const T & operator[]( unsigned int row, unsigned int col ) {
  const T* operator[]( unsigned int row ) {
	return  & results_on_host[ (row * NUM_COLUMNS) ];
  } // ()

  // -----------------------------------------------------------------
  // destructor
  // -----------------------------------------------------------------
  ~Results_Holder( ) {
	cudaFree( results_on_host );
	cudaFree( results_on_device );
	printf( " results memory (host and device) freed \n" );
  } // ()

  // -----------------------------------------------------------------
  // constructor
  // -----------------------------------------------------------------
  Results_Holder( unsigned int num_rows, unsigned int num_columns )
	: NUM_ROWS( num_rows ), NUM_COLUMNS( num_columns )
  {
	//
	// Get memory on the host.
	//
	results_on_host = my_malloc_2D< T >( NUM_ROWS, NUM_COLUMNS );
  
	//
	// Get memory on the device. Regular memory I guess, i.e. not a texture.
	//
	// Right now: I don't the differences between cudaMalloc and
	// cudaMallocManaged.
	//
	cudaMallocManaged( & results_on_device,
					   NUM_ROWS * NUM_COLUMNS * sizeof( T )
					   );

	check_cuda_call( " Results_Holder: cudaMallocManaged()" );
  } // ()

  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
  void copy_results_device_to_host() {
	cudaMemcpy( results_on_host,
				results_on_device,
				NUM_ROWS * NUM_COLUMNS * sizeof( T ),
				cudaMemcpyDeviceToHost );
	check_cuda_call( " copy_results_device_to_host " );
  } // ()

  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
}; // class

// ===================================================================
// Utility class for allocating memory on the device 
// binding it to a texture and copying the input data on the host
// to it.
// ===================================================================
template<typename T>
class Texture_Memory_Holder {
private:
  const unsigned int NUM_ROWS;
  const unsigned int NUM_COLUMNS;
public:
  
  cudaChannelFormatDesc channel_desc;

  T* data_on_device;
  
  cudaTextureObject_t texture;
  cudaResourceDesc resource_desc;
  cudaTextureDesc texture_desc;

  // -----------------------------------------------------------------
  // destructor
  // -----------------------------------------------------------------
  ~Texture_Memory_Holder( ) {
	cudaFree( data_on_device );
	cudaDestroyTextureObject( texture );
	printf( " data_on_device and texture memory freed \n" );
  } // ()

  // -----------------------------------------------------------------
  // constructor
  // -----------------------------------------------------------------
  Texture_Memory_Holder(
						Element_Type (*p_data)[],
						unsigned int num_rows,
						unsigned int num_columns
						)
	: NUM_ROWS( num_rows ), NUM_COLUMNS( num_columns )
  {

	//
	// get memory on the GPU to place our data
	//

	size_t total_size = NUM_ROWS * NUM_COLUMNS * sizeof( T ); 

	cudaMalloc( & data_on_device, total_size );

	check_cuda_call( " Texture_Memory_Holder: cudaMalloc() " );

	//printf( " Texture_Memory_Holder: element_type_size=%d, rows=%d, cols=%d, total_size=%zu\n", 
	//sizeof( Element_Type ), NUM_ROWS, NUM_COLUMNS, total_size );

	//
	// copy the data from here to the memory on the device
	//
	cudaMemcpy( data_on_device, // destination
				p_data, // source
				total_size, // size
				cudaMemcpyHostToDevice );

	check_cuda_call( " Texture_Memory_Holder: cudaMemcpy() " );

	//
	// create a channel.  What is this for?
	//
	channel_desc = cudaCreateChannelDesc< Element_Type >();

	//cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );

	check_cuda_call( " Texture_Memory_Holder: cudaCreateChannelDesc() " );

	//
	// create and configure a texture
	//
	memset( & resource_desc, 0, sizeof( cudaResourceDesc ) );

	resource_desc.resType = cudaResourceTypePitch2D;

	resource_desc.res.pitch2D.devPtr = data_on_device;

	resource_desc.res.pitch2D.width = NUM_COLUMNS;
	resource_desc.res.pitch2D.height = NUM_ROWS;

	resource_desc.res.pitch2D.desc = channel_desc;

	resource_desc.res.pitch2D.pitchInBytes = NUM_COLUMNS * sizeof( Element_Type );

	//
	//
	//
	memset( & texture_desc, 0, sizeof( cudaTextureDesc ) );

	// Last time I set this. Why?
	/*
	  texture_desc.normalizedCoords = false;  
	  texture_desc.readMode = cudaReadModeElementType;
	*/

	// Here it is where the texture is actually created
	cudaCreateTextureObject( & texture,
							 & resource_desc,
							 & texture_desc,
							 nullptr );

	check_cuda_call( " Texture_Memory_Holder:  cudaCreateTextureObject() " );
  } // ()

  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
};

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
  //return (T (*)[]) p_data;
  return static_cast< T (*)[] >( p_data );

} // ()

// ===================================================================
//
// main
//
// ===================================================================
int main( int n_args, char * args[] ) {

  printf( " starting \n" );

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
  printf( " %f \n", results[10][15] );
  printf( " %f \n", results[25][40] );
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
} // ()

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================

