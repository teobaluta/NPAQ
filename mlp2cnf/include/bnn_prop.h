#include "bnn.h"
#include "constraint_to_cnf.h"
#include <stdexcept>

extern int debug;

class BNNPropertyEncoder {
     private:
	vector<int> input_vars_bnn1;
	vector<int> output_vars_bnn1;
	vector<int> input_vars_bnn2;
	vector<int> output_vars_bnn2;
	int first_fresh_var;
	int biggest_aux_var;

	int at_most_k_different_inputs(int k, int first_fresh_var);
	int exactly_k_different_inputs(int k, int first_fresh_var);
	int outputs_are_different(int first_fresh_var);
	int inputs_are_different(int first_fresh_var);
	// if which_bnn = 0 then output of bnn1, else output of bnn2
	void output_is_equal_to(int which_bnn, int label_idx, bool value);
	// if which_bnn = 0 then input of bnn1, else output of bnn2
	void input_is_equal_to(int which_bnn, vector<bool> concrete_input);
	int sum_input_eq_1(bool which_bnn, int first_fresh_var);
	void input_bit_eq_to(int bit_idx, int value);

     public:
	BNNModel *bnn1;
	BNNModel *bnn2;
	string out_dir;
	string model_dir;
	string out_filename;
	vector<vector<int>> cnf;
	vector<vector<int>> xors;
	BNNPropertyEncoder(string bnn1_model_dir, string bnn2_model_dir, string out_filename, bool tandem=false);

	void encode_label(int label, vector<tuple<int, bool>> ip_constraints=vector<tuple<int, bool>>());
	void encode_dissimilarity(vector<tuple<int, bool>> ip_constraints=vector<tuple<int, bool>>());
	void encode_robustness(vector<bool> concrete_input, int perturb, bool equal);
	void encode_dp();
	void encode_fairness2(vector<int> locations, vector<bool> values1, vector<bool> values2, vector<vector<int>> dataset_constraints);
	void encode_fairness(vector<int> locations, vector<bool> values1, vector<bool> values2, vector<vector<int>> dataset_constraints);
	void encode_canary(vector<bool> canary, int non_canary_size);
	void write_to_file(int which_proj_bnn = 0, int which_bnn = 2);
};


class DifferentialNN {
     public:
	BNNModel *bnn1;
	BNNModel *bnn2;
	string out_dir;
	string model_dir;
	string out_filename;

	DifferentialNN(string bnn1_model_dir, string bnn2_model_dir, string out_dir);

	int encode();
};

