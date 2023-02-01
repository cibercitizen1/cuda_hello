// ===================================================================
//
// kernel
//
// ===================================================================

#define WHAT_KERNEL_TO_RUN kernel_detect

__global__ void kernel_detect( cuda4bytes * p_results, // -> results
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
  input_pixel.r = 128;
  input_pixel.g = 128;
  
  //
  // copy the new value to results
  //
  p_results[ (width * y_row) + x_column ] = my_rgba_to_cuda4bytes( input_pixel );

  //
  // control test
  //
  if (x_column == y_row || x_column == width-1 ) {
	cuda4bytes uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }
	
} // ()