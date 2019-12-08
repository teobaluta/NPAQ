#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "pb2cnf.h"
#include "utils.h"
#include "dirent.h"

using namespace std;
using namespace PBLib;

int debug = 1;

string OUTPUT_FILE = "cnf.dimacs";
string OUT_D_IJ = "out_dij";
string VARS = "metainfo";

vector<string> filename(string path, int number, bool internal_blk = true) {
	char separator;
#ifdef _WIN32
	separator = '\\';
#else
	separator = '/';
#endif
	string suffix;
	vector<string> files;
	if (internal_blk == true) {
		suffix = path + separator + "blk"  + to_string(number) + separator;
		files.push_back(suffix + "lin_weight.csv");
		files.push_back(suffix + "lin_bias.csv");
		files.push_back(suffix + "bn_weight.csv");
		files.push_back(suffix + "bn_bias.csv");
		files.push_back(suffix + "bn_mean.csv");
		files.push_back(suffix + "bn_var.csv");
	} else {
		suffix = path + separator + "out_blk" + separator;
		files.push_back(suffix + "lin_weight.csv");
		files.push_back(suffix + "lin_bias.csv");
	}

	return files;
}

int write_to_meta(string out_dir, string what, std::ios_base::openmode mode=std::ofstream::out) {
	ofstream meta_file;
	string meta_filename(out_dir + "/" + VARS);
	meta_file.open(meta_filename, mode);
	if (!meta_file.is_open()) {
		cout << "Error opening meta output file " << meta_filename << endl;
		return 1;
	}

	meta_file << what << endl;
	meta_file.close();
}

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

	BNNBlock(vector<string> files) {
		this->files = files;
		lin_weight = parseCSV(this->files[0]);
		lin_bias = parseCSV(this->files[1]);
		bn_weight = parseCSV(this->files[2]);
		bn_bias = parseCSV(this->files[3]);
		bn_mean = parseCSV(this->files[4]);
		bn_var = parseCSV(this->files[5]);

		in_size = lin_weight.size();
		out_size = lin_weight[0].size();
	}

	int encode(int first_fresh_var, int in_var_start, int out_var_start = 0) {
		PBConfig config = make_shared<PBConfigClass>();
		PB2CNF pb2cnf(config);

		int y;
		if (out_var_start == 0)
			this->out_var_start = in_var_start + lin_weight.size();
		else
			this->out_var_start = out_var_start;

		cout << "in_var_start=" << in_var_start << "; out_var_start=" << out_var_start << endl;
		for (int j = 0; j < lin_weight[0].size(); j++) {
			vector<WeightedLit> col;

			y = this->out_var_start + j;

			int w_minus = 0;
			int sum_aij = 0;
			for (int i = 0; i < lin_weight.size(); i++) {
				/*if (lin_weight[i][j] == 1) {*/
					//WeightedLit x = WeightedLit(i + in_var_start, 1);
					//col.push_back(x);
				//} else if (lin_weight[i][j] == -1) {
					//WeightedLit x = WeightedLit(-(i + in_var_start), 1);
					//col.push_back(x);
				/*}*/

				col.push_back(WeightedLit(i + in_var_start, lin_weight[i][j]));

				if (lin_weight[i][j] == -1)
					// w_minus += abs(round(lin_weight[i][j]));
					w_minus++;
				sum_aij += lin_weight[i][j];
			}
			this->constraints.push_back(col);

			if (bn_weight[0][j] == 0) {
				cout << bn_weight[0][j];
				cout << "DIV by 0" << endl;
				return 1;
			}
			float c = -(sqrt(bn_var[0][j]) / bn_weight[0][j]) * bn_bias[0][j] + bn_mean[0][j] - lin_bias[0][j];

			cout << "alfa = " << bn_weight[0][j] << endl;
			cout << "c = "  << c << endl;
			if (bn_weight[0][j] > 0) {
				c = ceil(c);
				//c = ceil(c / 2 +  sum_aij / 2) + w_minus;
				c = ceil((c + sum_aij) / 2);
				PBConstraint pbconstraint(col, GEQ, c);
				pbconstraint.addConditional(y);

				if (debug)
					pbconstraint.print(false);
				VectorClauseDatabase formula(config);
				AuxVarManager auxvars(first_fresh_var);

				pb2cnf.encode(pbconstraint, formula, auxvars);
				first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;
				if (debug == 2)
					for (auto clause : formula.getClauses()) {
						for (auto lit : clause) {
							cout << lit << " ";
						}
						cout << "0" << endl;
					}

				for (auto clause : formula.getClauses())
					this->cnf_formula.push_back(clause);

				PBConstraint pbct(col, LEQ, c - 1);
				pbct.addConditional(-y);

				if (debug)
					pbct.print(false);
				VectorClauseDatabase f(config);
				AuxVarManager auxv(first_fresh_var);

				pb2cnf.encode(pbct, f, auxv);
				first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;
				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

				for (auto clause : f.getClauses())
					this->cnf_formula.push_back(clause);
			} else {
				c = floor(c);
				c = floor(c / 2 +  sum_aij / 2);
				PBConstraint pbconstraint(col, LEQ, c);
				pbconstraint.addConditional(y);

				if (debug)
					pbconstraint.print(false);
				VectorClauseDatabase formula(config);
				AuxVarManager auxvars(first_fresh_var);

				pb2cnf.encode(pbconstraint, formula, auxvars);
				first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;
				if (debug == 2)
					for (auto clause : formula.getClauses()) {
						for (auto lit : clause) {
							cout << lit << " ";
						}
						cout << "0" << endl;
					}

				for (auto clause : formula.getClauses())
					this->cnf_formula.push_back(clause);

				PBConstraint pbct(col, GEQ, c + 1);
				pbct.addConditional(-y);

				if (debug)
					pbct.print(false);
				VectorClauseDatabase f(config);
				AuxVarManager auxv(first_fresh_var);

				pb2cnf.encode(pbct, f, auxv);
				first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;
				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

				for (auto clause : f.getClauses())
					this->cnf_formula.push_back(clause);
			}
		}
		this->out_var_end = y;

		if (debug) {
			cout << "Layer out_var [" << this->out_var_start << " - " << this->out_var_end << "]" << endl;
			cout << "Return " << first_fresh_var << endl;
		}

		return first_fresh_var;
	}

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

	BNNOutBlock(string out_dir, vector<string> files) {
		this->files = files;
		this->out_dir = out_dir;

		lin_weight = parseCSV(files[0]);
		lin_bias = parseCSV(files[1]);

		in_size = lin_weight.size();
		out_size = lin_weight[0].size() * lin_weight[0].size() - lin_weight[0].size();
	}

	int encode(int first_fresh_var, int in_var_start, int out_var_start) {
		PBConfig config = make_shared<PBConfigClass>();
		PB2CNF pb2cnf(config);

		float c;

		// this->d_ij
		int d_ij = in_var_start + lin_weight.size();
		d_ij_start = d_ij;
		this->out_var_start = out_var_start;
		int o = out_var_start;

		cout << "Out block encoding: start from " << in_var_start << endl;

		//vector<vector<int>> lin_weight;
		//vector<int> l;
		//l.push_back(1);
		//l.push_back(-1);

		//lin_weight.push_back(l);
		//vector<int> l1;
		//l1.push_back(-1);
		//l1.push_back(1);

		//lin_weight.push_back(l1);

		//vector<vector<float>> lin_bias;
		//vector<float> ff;
		//ff.push_back(-0.5);
		//ff.push_back(0.2);
		/*lin_bias.push_back(ff);*/

		cout << lin_weight[0].size() << endl;
		cout << lin_weight.size() << endl;
		// concrete values
		//static const int arr[] = {0,0,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0};
		//vector<int> val_x (arr, arr + sizeof(arr) / sizeof(arr[0]) );

		//for (int i = 0; i < sizeof(arr) / sizeof(arr[0]); i++)
		//	if (val_x[i] == 0)
		//		val_x[i] = -1;

		for (int col_i = 0; col_i < lin_weight[0].size(); col_i++) {

			vector<WeightedLit> sum_d_ij;
			for (int col_j = 0; col_j < lin_weight[0].size(); col_j++) {
				vector<WeightedLit> col;

				if (col_i == col_j)
					continue;

				int sum_a_pi = 0;
				int sum_a_pj = 0;
				int w_minus = 0;
				for (int i = 0; i < lin_weight.size(); i++) {
					int a_pi = lin_weight[i][col_i];
					int a_pj = lin_weight[i][col_j];
					sum_a_pi += a_pi;
					sum_a_pj += a_pj;

					//col.push_back(WeightedLit(i + in_var_start, a_pi - a_pj));
					//if (a_pi == 1)
					//	col.push_back(WeightedLit(i + in_var_start, 1));
					//if (a_pi == -1)
					//	col.push_back(WeightedLit(i + in_var_start, -1));
					//if (a_pj == 1)
					//	col.push_back(WeightedLit(i + in_var_start, -1));
					//if (a_pj == -1)
					//	col.push_back(WeightedLit(i + in_var_start, 1));

					if (a_pi == 1 && a_pj == -1)
						col.push_back(WeightedLit(i + in_var_start, 1));
					else if (a_pi == -1 && a_pj == 1) {
						col.push_back(WeightedLit(-(i + in_var_start), 1));
						w_minus++;
					}
				}
				c = lin_bias[0][col_j] - lin_bias[0][col_i] + sum_a_pi - sum_a_pj;
				//cout << "c float " << c << endl;
				int lala = ceil(c / 2);
				//cout << "lala = " << lala << endl;
				c = ceil(lala / 2);
				//cout << "c = " << c << endl;
				//cout << "bj = " << lin_bias[0][col_j] << "; bi = " << lin_bias[0][col_i] << endl;
				//cout << "sum_a_pi = " << sum_a_pi << "; sum_a_pj = " << sum_a_pj << endl;
				c = c + w_minus;
				//c = ceil(c / 2) + w_minus;

				this->constraints.push_back(col);

				if (col_i > col_j)
					c += 1;

				PBConstraint pbconstraint(col, GEQ, c);
				pbconstraint.addConditional(d_ij);

				if (debug) {
					cout << "d[" << col_i << "][" << col_j << "]: ";
					pbconstraint.print(false);
				}
				VectorClauseDatabase formula(config);
				AuxVarManager auxvars(first_fresh_var);

				pb2cnf.encode(pbconstraint, formula, auxvars);
				first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;
				if (debug == 2)
					for (auto clause : formula.getClauses()) {
						for (auto lit : clause) {
							cout << lit << " ";
						}
						cout << "0" << endl;
					}

				for (auto clause : formula.getClauses())
					this->cnf_formula.push_back(clause);

				PBConstraint pbct(col, LEQ, c - 1);
				pbct.addConditional(-d_ij);

				if (debug) {
					cout << "d[" << col_i << "][" << col_j << "]: ";
					pbct.print(false);
				}

				VectorClauseDatabase f(config);
				AuxVarManager auxv(first_fresh_var);
				pb2cnf.encode(pbct, f, auxv);
				first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;

				for (auto clause : f.getClauses())
					this->cnf_formula.push_back(clause);

				if (debug)
					cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

				sum_d_ij.push_back(WeightedLit(d_ij, 1));
				d_ij += 1;
			}

			// instead of one output variable use 10 output
			// variables to select one of the classes
			// e.g. instead of o = 5 we have [0, 0, 0, 0, 1, 0,..,
			// 0] a vector of size lin_weight[0].size()
			PBConstraint pb_out_ct_geq(sum_d_ij, GEQ, lin_weight[0].size() - 1);
			pb_out_ct_geq.addConditional(o);

			VectorClauseDatabase formula1(config);
			AuxVarManager auxvars1(first_fresh_var);

			pb2cnf.encode(pb_out_ct_geq, formula1, auxvars1);
			first_fresh_var = auxvars1.getBiggestReturnedAuxVar() + 1;

			for (auto clause : formula1.getClauses())
				this->cnf_formula.push_back(clause);

			PBConstraint pb_out_ct_leq(sum_d_ij, LEQ, lin_weight[0].size() - 1);
			pb_out_ct_leq.addConditional(o);
			VectorClauseDatabase formula2(config);
			AuxVarManager auxvars2(first_fresh_var);

			pb2cnf.encode(pb_out_ct_leq, formula2, auxvars2);
			first_fresh_var = auxvars2.getBiggestReturnedAuxVar() + 1;

			for (auto clause : formula2.getClauses())
				this->cnf_formula.push_back(clause);

			if (debug) {
				pb_out_ct_geq.print(false);
				pb_out_ct_leq.print(false);
			}

			//PBConstraint pbo1(sum_d_ij, GEQ, lin_weight[0].size() + 1);
			//pbo1.addConditional(-o);

			//VectorClauseDatabase f1(config);
			//AuxVarManager auxv1(first_fresh_var);

			//pb2cnf.encode(pbo1, f1, auxv1);
			//first_fresh_var = auxv1.getBiggestReturnedAuxVar() + 1;

			//for (auto clause : f1.getClauses())
				//this->cnf_formula.push_back(clause);

			//pbo1.print(false);

			PBConstraint pbo2(sum_d_ij, LEQ, lin_weight[0].size() - 2);
			pbo2.addConditional(-o);

			VectorClauseDatabase f2(config);
			AuxVarManager auxv2(first_fresh_var);

			pb2cnf.encode(pbo2, f2, auxv2);
			first_fresh_var = auxv2.getBiggestReturnedAuxVar() + 1;

			for (auto clause : f2.getClauses())
				this->cnf_formula.push_back(clause);

			pbo2.print(false);

			this->out_var_end = o;
			o += 1;
		}

		d_ij_end = d_ij;

		if (debug) {
			cout << "Layer out_var [" << this->out_var_start << " - " << this->out_var_end << "]" << endl;
			cout << "Return " << first_fresh_var << endl;
		}
		return first_fresh_var;
	}
};

class BNNModel {
      public:
	vector<BNNBlock> blocks;
	BNNOutBlock out_blk;
	int num_internal_blocks;
	string out_dir;
	string model_dir;
	ofstream out_file;

	BNNModel(char* model_dir, char *out_dir) {
		DIR *dir;
		struct dirent *ent;
		num_internal_blocks = 0;

		this->out_dir = out_dir;
		this->model_dir = model_dir;

		if ((dir = opendir(model_dir)) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				string d_name(ent->d_name);

				cout << d_name << endl;
				if (d_name.compare(0, 3, "blk") == 0) {
					num_internal_blocks += 1;
					vector<string> files = filename(this->model_dir, num_internal_blocks);
					cout << "Blk1 : " << files << endl;
					BNNBlock blk(files);
					blocks.push_back(blk);
				}
			}
			closedir(dir);
		} else {
			/* could not open directory */
			perror ("");
			return;
		}
		BNNOutBlock out_blk(out_dir, filename(this->model_dir, 0, false));
		this->out_blk = out_blk;
	}

	int encode(int first_fresh_var = 0) {
		int in_var_start = 1;
		int biggest_aux_var;
		// layer 0 output variables start after the input and output
		// variables
		int out_var_start = blocks[0].in_size + out_blk.lin_weight[0].size() + 1;

		// reserve variables, first_fresh_var should be after the
		// in_vars of the first block + num_blocks 
		int fresh_var = out_var_start;

		for (int i = 0; i < num_internal_blocks; i++)
			fresh_var += blocks[i].out_size;

		fresh_var += out_blk.out_size;

		if (first_fresh_var == 0)
			first_fresh_var = fresh_var;
		else if (first_fresh_var < fresh_var) {
			cout << "First fresh var should be greater than " << fresh_var << "!" << endl;
			cout << "Should use " << fresh_var << " value instead." << fresh_var << endl;
			return 1;
		}

		write_to_meta(this->out_dir, to_string(first_fresh_var));
		cout << "first_fresh_var = " << first_fresh_var << "; out var " << out_var_start << endl;

		for (int i = 0; i < num_internal_blocks; i++) {
			cout << "Encoding BLK" << i << endl;
			if (i == 0) {
				biggest_aux_var = blocks[i].encode(first_fresh_var, 1, out_var_start);
				write_to_meta(this->out_dir, to_string(1) + " " + \
					      to_string(blocks[i].out_var_start) + " " + \
					      to_string(blocks[i].out_var_end),
					      ios::app);
			} else {
				biggest_aux_var = blocks[i].encode(first_fresh_var, in_var_start);
				write_to_meta(this->out_dir, to_string(in_var_start) + " " + \
					      to_string(blocks[i].out_var_start) + " " + \
					      to_string(blocks[i].out_var_end),
					      ios::app);
			}
			in_var_start = blocks[i].out_var_start;
			first_fresh_var = biggest_aux_var;
		}

		// output variable 101 is reserved because it's an integer and
		// we convert the last constraint using z3
		biggest_aux_var = out_blk.encode(first_fresh_var, in_var_start, blocks[0].in_size + 1);
		write_to_meta(this->out_dir, to_string(in_var_start) + " " + \
			      to_string(out_blk.d_ij_start) + " " + \
			      to_string(out_blk.d_ij_end) + " " + \
			      to_string(out_blk.out_var_start) + " " + \
			      to_string(out_blk.out_var_end),
			      ios::app);

		ofstream out_file;
		string out_filename;
		out_filename = this->out_dir + "/" + OUTPUT_FILE;
		out_file.open(out_filename);
		if (out_file.is_open()) {
			cout << "Writing encoding to " << out_filename << endl;

			int total_clauses = 0;
			for (int i = 0; i < num_internal_blocks; i++) {
				total_clauses += blocks[i].cnf_formula.size();
			}

			total_clauses += out_blk.cnf_formula.size();

			out_file << "p cnf " << biggest_aux_var - 1 << " " << total_clauses << endl;

			for (int i = 0; i < num_internal_blocks; i++)
				for (auto clause : blocks[i].cnf_formula) {
					for (auto lit : clause)
						out_file << lit << " ";
					out_file << "0" << endl;
				}

			for (auto clause : out_blk.cnf_formula) {
				for (auto lit : clause)
					out_file << lit << " ";
				out_file << "0" << endl;
			}
			out_file.close();
			cout << "Finished writing CNF to " << out_filename << endl;
		} else {
			cout << "Error opening output file " << out_filename << endl;
			return 1;
		}

		return 0;
	}
};

int main(int argc, char *argv[]) {
	// ./bnn2cnf model_dir output_dir debug first_fresh_var

	if (argc < 2 || argc > 5) {
		cout << "./bnn2cnf model_dir output_dir debug first_fresh_var" << endl;
		return 1;
	}

	// default debug level is 1
	if (argc >= 4)
		debug = atoi(argv[3]);


	int ret_code;

	try {
		BNNModel model(argv[1], argv[2]);

		if (argc == 5) {
			int first_fresh_var = atoi(argv[4]);
			if (first_fresh_var < 0) {
				cout << "Fresh variables should be positive!" << endl;
				return -1;
			}
			ret_code = model.encode(first_fresh_var);
		} else
			ret_code = model.encode();
	} catch (int e) {
		cout << "Error " << e << endl;
	}

	return ret_code;
}
