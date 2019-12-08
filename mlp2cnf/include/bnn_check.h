#ifndef BNN_CHECK_H
#define BNN_CHECK_H

#include <fstream>
#include <iostream>
#include <vector>
#include "bnn.h"
#include <cryptominisat5/cryptominisat.h>
#include <assert.h>
using namespace CMSat;

int generate_models(string model_dir, string samples_fname);

#endif
