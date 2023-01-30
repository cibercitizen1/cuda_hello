//--------------------------------------------------------------
// util.cpp
//--------------------------------------------------------------

#include <util.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//...............................................................
void pause(const char * msg)  {
  string a; 
  cerr << "pause: " << msg << " "; 
  cin >> a;
}

//...............................................................
void stop_execution(const char * text) {
  printf ("%s\n", text);
  printf ("!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf (" program done ");
  printf ("!!!!!!!!!!!!!!!!!!!!!!!!\n");
  exit(0);
} // ()

//...............................................................
FILE * open_file_to_read(const char * name) {
  FILE * file_ref = fopen(name,"r"); 
  if (file_ref == nullptr) {
	throw Error_FileNotFound();
  }
  return file_ref;
} // ()


//...............................................................
FILE * open_file_to_write(const char * name) {
  FILE * file_ref;
  file_ref = fopen(name,"w"); 
  if (file_ref == 0) {
	throw Error_FileNotFound();
  }
  return file_ref;
} // ()

//...............................................................
void dir_contents (const char * dir, vector<string> &names)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir)) == NULL) {
		throw Error_DirectoryNotFound ();
    }

    while ((dirp = readdir(dp)) != NULL) {
	  string nombre(dirp->d_name);
	  if (nombre == "."  || nombre == "..") {
	  } else {
        names.push_back(nombre);
	  }
    }
    closedir(dp);
}

//...............................................................
void prefixed_dir_contents (const char * dir,
							vector<string> &names)
{
  vector<string> unprefixed;

  dir_contents(dir, unprefixed);

  for (int i=0; i<unprefixed.size(); i++) {
	names.push_back(string(dir)+ "/" + unprefixed[i]);
  }
}


// ..................................................................
// ..................................................................
double min_of( const double * samples, const unsigned int size)  
{
  double min = samples[0];
  for (unsigned int i=1; i<size; i++) {
	if (samples[i] < min) {
	  min = samples[i];
	}
  }
  return min;
} // ()

// ..................................................................
// ..................................................................
double max_of( const double * samples, const unsigned int size)  
{
  double max = samples[0];
  for (unsigned int i=1; i<size; i++) {
	if (samples[i] > max) {
	  max = samples[i];
	}
  }
  return max;
} // ()

// ..................................................................
// ..................................................................
void normalise_samples( double * samples, const unsigned int size) 
{
  // substract the minimum to all the values
  double min = min_of(samples, size);
  for (unsigned int i=0; i<size; i++) {
	samples[i] -= min;
  }
} // ()

// ..................................................................
// ..................................................................
void limit_samples(double * samples,
				   const double * const mins,
				   const double * const maxs,
				   const unsigned int size) 
{
  for (unsigned int i=0; i<size; i++) {
	if (samples[i] < mins[i]) {
	  samples[i] = mins[i];
	} else if (samples[i] > maxs[i]) {
	  samples[i] = maxs[i];
	}
  } // for
} // ()

// ..................................................................
// ..................................................................
double mean(const double * samples, const unsigned int size) 
{
  double sum = 0.0;
  for (unsigned int i=0; i<size; i++) {
	sum += samples[i];
  }
  return sum/size;
}

// ..................................................................
// ..................................................................
double standard_deviation (const double * samples,
						   const unsigned int size) 
{
  double m = mean(samples, size);
  double sum = 0.0;
  
  for (unsigned int i=0; i<size; i++) {
	double dif = samples[i] - m;
	sum +=  (dif * dif);
  } // for
  
  return sqrt(sum) / (size-1);
}

// ..................................................................
// ..................................................................
void print_array_double (const double * array, const unsigned int size)
{
  for (unsigned int i=0; i<size; i++) {
		  printf ("%.4f ", array[i]);
  } // for
  printf ("\n");
}

// ..................................................................
// ..................................................................
int search_in_byte_array(const unsigned char * array,
						  const unsigned int array_length,
						  const unsigned char * wanted,
						  const unsigned int wanted_length)
{

  int i;
  int j;

  for (i=0; i<=array_length-wanted_length; i++) {
	// printf ("\ni = %d\n", i);
	for (j=0; j<wanted_length; j++) {
	  // printf ("   j = %d ", j);
	  // printf (" %d == %d ", array[i+j], wanted[j]);
	  if (array[i+j] != wanted[j]) {
		break;
	  }
	} // for
	if (j==wanted_length) {
	  return i;
	}
  } // for
  return -1;
} // ()


// ..................................................................
// ..................................................................
int search_in_file(FILE * file_ref,
				   const unsigned char * wanted, 
				   const unsigned int wanted_length)
{

  rewind(file_ref); // make sure we start at the beginning

  unsigned int wanted_pos = 0;

  unsigned char read_byte;

  // en principio, mientras no se acabe el file_refero
  while ( ! feof(file_ref) ) {
	fread ( & read_byte, 1, 1, file_ref);
	if (wanted[wanted_pos] == read_byte) {
	  // coincide el byte leído
	  wanted_pos++;
	} else {
	  // NO coincide el byte leído
	  wanted_pos = 0;
	}

	if (wanted_pos==wanted_length) {
	  // lo encontré
	  return ftell(file_ref)-wanted_length;
	}

  } // while
  
  return -1; // NO LO ENCONTRE
} // ()

// ..................................................................
// ..................................................................
void date_and_time(char * date, char * hour) {

	time_t moment = time(0);
	// struct tm * timeStruct = localtime(&tiempo);
	struct tm * timeStruct = gmtime(&moment);

	sprintf (&date[0], "%.4d%.2d%.2d", timeStruct->tm_year+1900, timeStruct->tm_mon+1, timeStruct->tm_mday);

	sprintf (&hour[0], "%.2d%.2d%.2d", timeStruct->tm_hour, timeStruct->tm_min, timeStruct->tm_sec);

}

//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
