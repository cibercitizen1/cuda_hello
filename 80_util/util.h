//--------------------------------------------------------------
// util.h
//--------------------------------------------------------------

#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>

#include <iostream>
using namespace std;

// ..................................................................
class Error_Malloc { };
class Error_FileNotFound { };
class Error_DirectoryNotFound { };
class Error_ReadingFile { };

// ..................................................................
template<typename T>
T my_malloc( const long unsigned size )
{
  void * ptr = malloc( size );

  assert( ptr != nullptr && "my_malloc failed" );

  return (T) ptr;
} // ()

// ..................................................................
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
/*
//auto kk = new int [10][20];
// OK int (* kk)[20] = new int [10][20];
int (* kk)[20] = new int [10][20];
kk[9][2] = 13;
*/

// ..................................................................
template<typename T>
FILE * open_file_to_read(const char * name);

// ..................................................................
FILE * open_file_to_write(const char * name);

// ..................................................................
void dir_contents (const char * dir, vector<string> &names);

// ..................................................................
void prefixed_dir_contents (const char * dir,
							vector<string> &names);

// ..................................................................
template <class T>
void  traza(T msg) {
  cerr << msg << endl;
  cerr << flush;
}

// ..................................................................
void pause(const char * msg);

// ..................................................................
void print_array_double (const double * array, const unsigned int size); 

// ..................................................................
double mean(const double * samples, const unsigned int size);

// ..................................................................
double standard_deviation(const double * samples,
						 const unsigned int size);

// ..................................................................
double min_f( const double * samples, const unsigned int size);

// ..................................................................
double max_of( const double * samples, const unsigned int size);

// ..................................................................
void normalise_samples( double * samples, const unsigned int size);

// ..................................................................
void limit_samples(double * samples,
				   const double * const mins,
				   const double * const maxs,
				   const unsigned int size); 

// ..................................................................
void stop_execution(const char * text);

// ..................................................................
template<class T>
void print_array(const T * array, const unsigned int size) 
{
  for (unsigned int i=0; i<size; i++) {
	cout << array[i] << ", ";
	// printf ("%.12f, ", array[i]);
  }
  cout << endl;
}

// ..................................................................
// fecha 20120318 => 8 chars aaaammdd
// hora: 213032 => 6 char hhmmss
void date_and_time (char * date, char * hour); 

// ..................................................................
template<class T>
unsigned int copiaBytes(const unsigned char * array, const unsigned int pos, T * dest) {
  const T * p = (const T *) & array[pos];
  
  (*dest) = (*p);
  // así no es: (*dest) = array[pos];
  return pos + sizeof(T);
}

// ..................................................................
int search_byte_array(const unsigned char * array,
					  const unsigned int array_length,
					  const unsigned char * wanted,
					  const unsigned int wanted_length);

// ..................................................................
// returns the position (offset) where 'wanted' bytes are,
// but the file indicator is rewinded
//
// returns -1 if not found
// ..................................................................
int search_in_file(FILE * file_ref,
				   const unsigned char * wanted, 
				   const unsigned int wanted_length);

// ..................................................................
// ..................................................................
inline void original_vs_estimated_differences (
								const double * original_samples,
								const double * estimated_samples,
								const unsigned int size,
								double * differences, // out
								double * squared_differences // out
								)
{

  (*differences) = 0.0;
  (*squared_differences) = 0.0;

  for (unsigned int i=0; i<size; i++) {
	
	double dif = original_samples[i]-estimated_samples[i];
	(*differences) += dif;
	(*squared_differences) += dif*dif;

  } // for

} // ()
  
#endif
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
