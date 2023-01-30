//--------------------------------------------------------------
// CommandLineParser.h
//--------------------------------------------------------------

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <map>
#include <string>
#include <iostream>
#include <sstream>

#include <util.h>

#include <string.h>
#include <stdio.h>

using namespace std;

//..........................................................................
//..........................................................................
class Command_Line_Parser {

 private:

  typedef map<string, string> Map_StringString;
  // typedef Map_StringString::iterator Map_Iterator;
  typedef Map_StringString::const_iterator Map_Iterator;

  Map_StringString the_options;
  
  const Map_Iterator map_end;

 public:

  //..............................................................
  //..............................................................
  Command_Line_Parser (int nArgs, char * args[], const string & default_values = "") 
	: map_end(the_options.end())
  {
	char * line =  my_malloc<char*>(default_values.size()); // must copy for strtok to work
	sprintf (line, "%s", default_values.c_str());

	// first, split the default values string and add each piece
	char * trozo = 0;
	trozo = strtok(line, " \t");

	while ( trozo != 0 ) {
	   add_option(trozo);
	  trozo = strtok(0, " \t");
	}

	// then, add the words in the command line 
	for (int i = 1; i<= nArgs-1; i++) {
	  add_option( args[i] );
	}  // for

	free(line);
	  
  } // ()

  //..............................................................
  //..............................................................
  void remove_option(const string & option) {
	the_options.erase (option);
  } // 

  //..............................................................
  //..............................................................
  void add_option(const string & option) {
	// cout << " anyadiendo " << option << endl;
	  size_t eq_index = option.find("=");

	  if (eq_index==string::npos) {
		// without equal sign
		the_options[option] = "";
	  } else {
		// with equal sign
		string key = option.substr(0, eq_index);
		string value = option.substr(eq_index+1);

		the_options[key] = value;
	  }
  }

  //..............................................................
  //..............................................................
  bool option_exists(const string & key) const {
	return the_options.find(key) != map_end;
  }

  //..............................................................
  //..............................................................
  template<typename T>
  T option_value(const string & key) {

	T result;
	if (option_exists(key)) {
	  istringstream iss (the_options[key]);
	  iss >> result;
	}

	return result;
  }

  //..............................................................
  //..............................................................
  void print() const {
	Map_Iterator it;
	for ( it=the_options.begin() ; it != the_options.end(); it++ ) {
	  cout << (*it).first << " => " << (*it).second << endl;
	}
  }

  //..............................................................
  //..............................................................
  string aString () const {
	ostringstream oss;
	Map_Iterator it;
	for ( it=the_options.begin() ; it != the_options.end(); it++ ) {
	  oss << (*it).first << "=" << (*it).second << " ";
	}
	// oss << endl;
	
	return oss.str();
  } // ()

}; // class

#endif
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
