#ifndef CONSTRAINT_TO_CNF_H
#define CONSTRAINT_TO_CNF_H

#include <vector>
#include <iostream>
#include "pb2cnf.h"
#include "math.h"

#include "err.h"

using namespace std;
using namespace PBLib;

extern int debug;

// Implementation of transformation of logical properties to CNF.
//
// FIXME Consider using something like z3 to encode these things.
// These are basic constraints that can be composed and added over the BNNs.

// TODO add a function that just encodes xor constraint?
int add_xor_directly(vector<int> bit_vec1, vector<int> bit_vec2,
		     vector<vector<int>> &xors, vector<int> &aux_vars,
		     int first_fresh_var);
int add_xor_constraint(vector<int> bit_vec1, vector<int> bit_vec2,
		       vector<vector<int>> &cnf, vector<int> &aux_vars,
		       int first_fresh_var);

int add_at_most_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf);

int add_at_most_bin_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf);

int add_at_least_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf);

int add_equals_to(vector<int> bit_vec, vector<bool> values, vector<vector<int>> &cnf);

void add_equals_to(vector<int> bit_vec, int idx, bool value, vector<vector<int>> &cnf);

int add_equals_to(vector<int> bit_vec1, vector<int> bit_vec2,  vector<vector<int>> &cnf);

#endif /* CONSTRAINT_TO_CNF_H */
