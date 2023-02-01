// -*- mode: c++ -*-
// ===================================================================
//
// kernel
//
// ===================================================================

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

  // read the eight neighbour pixels
  PixelRGBA neighbour_pixels[8];

  neighbour_pixels[0] = access_pixel( in_data_texture, x_column-1, y_row-1 );
  neighbour_pixels[1] = access_pixel( in_data_texture, x_column, y_row-1 );
  neighbour_pixels[2] = access_pixel( in_data_texture, x_column+1, y_row-1 );

  neighbour_pixels[3] = access_pixel( in_data_texture, x_column-1, y_row );
  neighbour_pixels[4] = access_pixel( in_data_texture, x_column+1, y_row );

  neighbour_pixels[5] = access_pixel( in_data_texture, x_column-1, y_row+1 );
  neighbour_pixels[6] = access_pixel( in_data_texture, x_column, y_row+1 );
  neighbour_pixels[7] = access_pixel( in_data_texture, x_column+1, y_row+1 );

  PixelRGBA new_pixel;
  mean_of_pixels( neighbour_pixels, 8, & new_pixel );

  //
  // do some processing
  // delete the blue value
  //
	input_pixel.r = 128;
  ///input_pixel.g = 128;
  
  //
  // copy the new value to results
  //
  p_results[ (width * y_row) + x_column ] =
	my_rgba_to_cuda4bytes( neighbour_pixels[0] );

  //
  // control test: draw a diagonal and column
  //
  if (x_column == y_row || x_column == width-1 ) {
	cuda4bytes uc4;
	uc4.x = x_column;
	uc4.y = y_row;
	p_results[ (width * y_row) + x_column ] = uc4;
  }
	
} // ()
// ===================================================================
// ===================================================================
// ===================================================================
// ===================================================================
