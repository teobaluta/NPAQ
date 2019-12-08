#ifndef BNN_H
#define BNN_H

#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <set>
#include <stdexcept>

#include "pb2cnf.h"
#include "utils.h"
#include "dirent.h"
#include "err.h"

using namespace std;
using namespace PBLib;

extern int debug;
extern string encoder;


class BNNBlock {
      public:
	vector<vector<WeightedLit>> constraints;
	vector<string> files;
	vector<vector<int>> cnf_formula;
	int out_var_end;
	int out_var_start;
	vector<vector<float>> lin_weight;
	vector<vector<float>> lin_bias;
	vector<vector<float>> bn_weight;
	vector<vector<float>> bn_bias;
	vector<vector<float>> bn_mean;
	vector<vector<float>> bn_var;
	int in_size;
	int out_size;

	BNNBlock(vector<string> files);

	int encode(int first_fresh_var, int in_var_start, int out_var_start = 0, set<int> perturb = {});
};

class BNNOutBlock {
      public:
	vector<vector<WeightedLit>> constraints;
	vector<string> files;
	vector<vector<int>> cnf_formula;
	int out_var_end;
	int out_var_start;
	string out_dir;
	int in_size;
	int out_size;
	vector<vector<float>> lin_weight;
	vector<vector<float>> lin_bias;
	int d_ij_start;
	int d_ij_end;

	BNNOutBlock() {}

	BNNOutBlock(string out_dir, vector<string> files);

	int encode(int first_fresh_var, int in_var_start, int out_var_start);
};

class BNNModel {
      public:
	vector<BNNBlock> blocks;
	BNNOutBlock out_blk;
	int num_internal_blocks;
	string out_dir;
	string model_dir;
	ofstream out_file;
	int input_size;
	int output_size;

	set<int> perturb;

	BNNModel() {}

	BNNModel(string model_dir, string out_dir, set<int> perturb = {});

	virtual int encode(int first_fresh_var = 0, int in_var_start = 1, int out_var_start = 0);
	int encode_internal(int first_fresh_var = 0, int in_var_start = 1, int out_var_start = 0);
	void write_to_file(ofstream& out_file);
};

AMK_ENCODER::PB2CNF_AMK_Encoder get_enc_type();
#endif /* BNN_H */
