
#include <CommandLineParser.h>
#include <iostream>

using namespace std;

// ..................................................................
// ..................................................................
int main(int argc, char* args[] )
{

  cout << " ------- starting -------- " << endl;

  Command_Line_Parser clp (argc, args, "images=../../default_dat directory=default/something");

  cout << clp.option_exists("gnuplot") << endl;

  cout << clp.option_value<double>("threshold") << endl;
  cout << clp.option_value<int>("threshold") << endl;
  cout << clp.option_value<string>("images") << endl;
  cout << clp.option_value<string>("directory") << endl;

  cout << " ------- done -------- " << endl;
}
