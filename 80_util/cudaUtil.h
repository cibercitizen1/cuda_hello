// -*- mode: c++ -*-
// ===================================================================
// cudaUtil.h
// ===================================================================
#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdio.h>
#include <assert.h>

// ===================================================================
// ===================================================================
void check_cuda_call( const char * msg );

// ===================================================================
// Let's use cudaMallocHost()
// which gets "pinned memory" in the CPU for us.
// I guess that means that the memory is aligned so that transfers
// from and to the GPU are faster.
// ===================================================================
template<typename T>
T * my_malloc( unsigned int NUM_ROWS, unsigned int NUM_COLUMNS=1, unsigned int DIM_Z=1) {
  
  //
  // compute the size required
  //
  size_t size = NUM_ROWS * NUM_COLUMNS * DIM_Z * sizeof( T );

  printf( " my_malloc: size needed = %zu \n", size );

  //
  // malloc
  //
  T* ptr = nullptr;
  cudaMallocHost( & ptr, size );

  check_cuda_call( "my_malloc(): cudaMallocHost()" );
  
  //
  // make sure we've got memory
  //
  assert( ptr != nullptr && "my_malloc failed" );

  //return ( T * [] ) ptr;
  return ptr;
} // ()

// ===================================================================
// Utility class for allocating memory both on the device
// and on the host.
// ===================================================================
template<typename Type>
class Results_Holder {
private:
  const unsigned int NUM_ROWS;
  const unsigned int NUM_COLUMNS;
public:
  Type * results_on_host;
  Type * results_on_device;

  // -----------------------------------------------------------------
  // Used to access to the correct row of results_on_host.
  // Column dimension is required to get the correct one.
  // Because a pointer is returned, [] can be chained:
  // Example:
  // results[10][15]
  // -----------------------------------------------------------------
  const Type& operator()( unsigned int row, unsigned int col ) {
	return  results_on_host[ (row * NUM_COLUMNS) + col ];
  } // ()

  // -----------------------------------------------------------------
  const Type* operator[]( unsigned int row ) {
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
	results_on_host = my_malloc< Type >( NUM_ROWS, NUM_COLUMNS );
  
	//
	// Get memory on the device. Regular memory I guess, i.e. not a texture.
	//
	// Right now: I don't the differences between cudaMalloc and
	// cudaMallocManaged.
	//
	cudaMallocManaged( & results_on_device,
					   NUM_ROWS * NUM_COLUMNS * sizeof( Type )
					   );

	check_cuda_call( " Results_Holder: cudaMallocManaged()" );
  } // ()

  // -----------------------------------------------------------------
  // -----------------------------------------------------------------
  void copy_results_device_to_host() {
	cudaMemcpy( results_on_host,
				results_on_device,
				NUM_ROWS * NUM_COLUMNS * sizeof( Type ),
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
template<typename Type>
class Texture_Memory_Holder {
private:
  const unsigned int NUM_ROWS;
  const unsigned int NUM_COLUMNS;
public:
  
  cudaChannelFormatDesc channel_desc;

  Type* data_on_device;
  
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
						Type *p_data,
						unsigned int num_rows,
						unsigned int num_columns
						)
	: NUM_ROWS( num_rows ), NUM_COLUMNS( num_columns )
  {

	//
	// get memory on the GPU to place our data
	//

	size_t total_size = NUM_ROWS * NUM_COLUMNS * sizeof( Type ); 

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
	channel_desc = cudaCreateChannelDesc< Type >();

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

	resource_desc.res.pitch2D.pitchInBytes = NUM_COLUMNS * sizeof( Type );

	//
	//
	//
	memset( & texture_desc, 0, sizeof( cudaTextureDesc ) );

	// Last time I set this. Why?
  texture_desc.normalizedCoords = false;  
  texture_desc.readMode = cudaReadModeElementType;
  /*
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
// ===================================================================
inline int cudaGetMaxGflopsDeviceId() {
  int device_count = 0;
  cudaGetDeviceCount( &device_count );

  cudaDeviceProp device_properties;
  int max_gflops_device = 0;
  int max_gflops = 0;

  int current_device = 0;
  cudaGetDeviceProperties( &device_properties, current_device );
  max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
  ++current_device;

  while( current_device < device_count ) {
    cudaGetDeviceProperties( &device_properties, current_device );
    int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
    if( gflops > max_gflops )
    {
      max_gflops        = gflops;
      max_gflops_device = current_device;
    }
    ++current_device;
  } // while

  return max_gflops_device;
}


// ===================================================================
// ===================================================================
#endif

// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
