#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>

using namespace std;

vector<vector<float>> parseCSV(string filename, char delimiter = ',', int debug = 0);
int ensure_dir(char *path);

vector<string> filename(string path, int number, bool internal_blk = true);

int write_to_meta(string out_dir, string what, std::ios_base::openmode mode=std::ofstream::out);
#endif
