#include "bnn_prop.h"

// FIXME RETURN CODES FOR ALL OF THE C++ CODE SOME COMMON.H
BNNPropertyEncoder::BNNPropertyEncoder(string bnn1_model_dir, string bnn2_model_dir,
				       string out_filename, bool tandem) {
	this->model_dir = model_dir;

	this->out_filename = out_filename;

	this->bnn1 = new BNNModel(bnn1_model_dir, "");
	this->bnn2 = new BNNModel(bnn2_model_dir, "");

	cout << "BNN 1 in_size=" << this->bnn1->input_size << "; out_size=" 
		<< this->bnn1->output_size << endl;
	cout << "BNN 2 in_size=" << this->bnn2->input_size << "; out_size="
		<< this->bnn2->output_size << endl;
	if (tandem) {
		first_fresh_var = 1;
		int in_var_start_bnn1 = 1;

		first_fresh_var = bnn1->encode_internal(0, in_var_start_bnn1);
		cout << "first_fresh after bnn1 internal block encoding " << first_fresh_var << endl;
		for (int i = 1; i <= bnn1->input_size; i++) {
			if (debug)
				cout << "adding " << i << " to bnn1 inp vars" << endl;
			this->input_vars_bnn1.push_back(i);
		}

		int in_var_start_bnn2 = first_fresh_var;
		cout << "Start BNN2 at = " << in_var_start_bnn2 << endl;

		for (int i = 0; i < bnn2->input_size; i++) {
			if (debug)
				cout << "adding " << i + in_var_start_bnn2 << " to bnn2 inp vars" << endl;
			this->input_vars_bnn2.push_back(i + in_var_start_bnn2);
		}

		first_fresh_var = bnn2->encode_internal(0, in_var_start_bnn2);
		cout << "first_fresh after bnn2 internal block encoding " << first_fresh_var << endl;
	} else {
		// first_fresh_var is class private field; it keeps track of the number of auxiliary variables introduced by encoding the two neural networks
		// adding constraints over the encoding
		// of the two neural nets
		this->first_fresh_var = 1;
		int in_var_start_bnn1 = 1;
		int i = 1;

		for (i = 1; i <= bnn1->input_size; i++) {
			if (debug)
				cout << "adding " << i << " to bnn1 inp vars" << endl;
			this->input_vars_bnn1.push_back(i);
		}

		// when first_fresh_var = 0 the encode method will compute the
		// first_fresh_var
		first_fresh_var = bnn1->encode(0, in_var_start_bnn1);

		i = bnn1->out_blk.out_var_start;
		while (i <= bnn1->out_blk.out_var_end) {
			if (debug)
				cout << "adding " << i << " to bnn2 out vars" << endl;
			this->output_vars_bnn1.push_back(i);
			i++;
		}

		int in_var_start_bnn2 = first_fresh_var;
		cout << "Start BNN2 at = " << in_var_start_bnn2 << endl;

		for (i = 0; i < bnn2->input_size; i++) {
			if (debug)
				cout << "adding " << i + in_var_start_bnn2 << " to bnn2 inp vars" << endl;
			this->input_vars_bnn2.push_back(i + in_var_start_bnn2);
		}

		// when first_fresh_var = 0 the encode method will compute the
		// first_fresh_var
		first_fresh_var = bnn2->encode(0, in_var_start_bnn2);

		i = bnn2->out_blk.out_var_start;
		while (i <= bnn2->out_blk.out_var_end) {
			if (debug)
				cout << "adding " << i << " to bnn2 out vars" << endl;
			this->output_vars_bnn2.push_back(i);
			i++;
		}
	}
}

/*
 * x1, x2, aux_vars bitvectors
 *
 * P2(x1, x2, k) = (aux_vars = x1 xor x2) and (sum(aux_vars) <= k and
 * sum(aux_vars) >= 1)
 */
int BNNPropertyEncoder::at_most_k_different_inputs(int k, int first_fresh_var) {
	vector<int> aux_vars;

	first_fresh_var = add_xor_directly(this->input_vars_bnn1, this->input_vars_bnn2, xors, aux_vars, first_fresh_var);
	if (debug)
		cout << "AUX VAR SIZE " << aux_vars.size() << endl;
	for (int i = 0; i < aux_vars.size(); i++) {
		cout << aux_vars[i] << " ";
	}
	cout << endl;

	if (debug) {
		cout << "XORS " << xors.size() << endl;
		for (int i = 0; i < xors.size(); i++) {
			for (auto lit : xors[i]) {
				cout << lit << " ";
			}
			cout << endl;
		}
	}
	cout << "first fresh after xor constraints " << first_fresh_var << endl;

	first_fresh_var = add_at_most_k(aux_vars, k, first_fresh_var, cnf);
	cout << "first fresh after at most k " << first_fresh_var << endl;

	first_fresh_var = add_at_least_k(aux_vars, 1, first_fresh_var, cnf);
	cout << "first fresh after at least 1 " << first_fresh_var << endl;

	return first_fresh_var;
}

int BNNPropertyEncoder::exactly_k_different_inputs(int k, int first_fresh_var) {
	vector<int> aux_vars;

	first_fresh_var = add_xor_directly(this->input_vars_bnn1, this->input_vars_bnn2, xors, aux_vars, first_fresh_var);
	if (debug)
		cout << "AUX VAR SIZE " << aux_vars.size() << endl;
	for (int i = 0; i < aux_vars.size(); i++) {
		cout << aux_vars[i] << " ";
	}
	cout << endl;

	if (debug) {
		cout << "XORS " << xors.size() << endl;
		for (int i = 0; i < xors.size(); i++) {
			for (auto lit : xors[i]) {
				cout << lit << " ";
			}
			cout << endl;
		}
	}
	cout << "first fresh after xor constraints " << first_fresh_var << endl;

	first_fresh_var = add_at_most_k(aux_vars, k, first_fresh_var, cnf);
	cout << "first fresh after at most k " << first_fresh_var << endl;

	first_fresh_var = add_at_least_k(aux_vars, k, first_fresh_var, cnf);
	cout << "first fresh after at least k " << first_fresh_var << endl;

	return first_fresh_var;
}

int BNNPropertyEncoder::outputs_are_different(int first_fresh_var) {
	vector<int> aux_vars;

	first_fresh_var = add_xor_constraint(this->output_vars_bnn1, this->output_vars_bnn2,
					     this->cnf, aux_vars, first_fresh_var);
	// and at least one of the aux_vars[i] = (x1[i] != x2[i]) is equal to 1
	vector<int> at_least_one_diff_clause;
	for (int i = 0; i < aux_vars.size(); i++) {
		at_least_one_diff_clause.push_back(aux_vars[i]);
	}
	cnf.push_back(at_least_one_diff_clause);

	return first_fresh_var;
}

int BNNPropertyEncoder::inputs_are_different(int first_fresh_var) {
	vector<int> aux_vars;

	first_fresh_var = add_xor_constraint(this->input_vars_bnn1, this->input_vars_bnn2,
					     this->cnf, aux_vars, first_fresh_var);
	// and at least one of the aux_vars[i] = (x1[i] != x2[i]) is equal to 1
       /* vector<int> at_least_one_diff_clause;*/
	//for (int i = 0; i < aux_vars.size(); i++) {
		//at_least_one_diff_clause.push_back(aux_vars[i]);
	//}
	//cnf.push_back(at_least_one_diff_clause);

	// all inputs are different
	cnf.push_back(aux_vars);

	return first_fresh_var;
}

void BNNPropertyEncoder::output_is_equal_to(int which_bnn, int label_idx, bool value) {
	if (which_bnn > 2)
		return;

	if (which_bnn == 0)
		add_equals_to(this->output_vars_bnn1, label_idx, value, this->cnf);
	else
		add_equals_to(this->output_vars_bnn2, label_idx, value, this->cnf);
}

void BNNPropertyEncoder::input_is_equal_to(int which_bnn, vector<bool> concrete_input) {

	// no extra variables are added
	if (which_bnn == 0)
		add_equals_to(this->input_vars_bnn1, concrete_input, this->cnf);
	else
		add_equals_to(this->input_vars_bnn2, concrete_input, this->cnf);
	
}

void BNNPropertyEncoder::encode_label(int label_idx, vector<tuple<int, bool>> ip_constraints) {
	if (ip_constraints.size() > 0) {
		vector<int> bit_vec;
		vector<bool> values;
		for (const auto &i: ip_constraints) {
			bit_vec.push_back(get<0>(i));
			values.push_back(get<1>(i));
			if (get<0>(i) < 0 or get<0>(i) > this->input_vars_bnn1.size()) {
				cout << "WARNING input constraints not on input vars";
				exit(EXIT_FAILURE);
			}
		}
		add_equals_to(bit_vec, values, this->cnf);
	}
	cout << "first fresh var after BNNs encoding " << first_fresh_var << endl;
	output_is_equal_to(0, label_idx, true);
	this->biggest_aux_var = first_fresh_var;

	write_to_file(0, 0);
}

void BNNPropertyEncoder::encode_dissimilarity(vector<tuple<int, bool>> ip_constraints) {
	if (ip_constraints.size() == 0) {
		cout << "No input constraints" << endl;
		// inputs of the two neural nets are the same
		add_equals_to(this->input_vars_bnn1, this->input_vars_bnn2, this->cnf);
		first_fresh_var = outputs_are_different(first_fresh_var);

		biggest_aux_var = first_fresh_var;
		write_to_file();
		return;
	}

	cout << "Encode dis-similarity with constraints" << endl;
	add_equals_to(this->input_vars_bnn1, this->input_vars_bnn2, this->cnf);
	first_fresh_var = outputs_are_different(first_fresh_var);

	biggest_aux_var = first_fresh_var;

	vector<int> bit_vec;
	vector<bool> values;
	for (const auto &i: ip_constraints) {
		bit_vec.push_back(get<0>(i));
		values.push_back(get<1>(i));
		if (get<0>(i) < 0 or get<0>(i) > this->input_vars_bnn1.size()) {
			cout << "WARNING input constraints not on input vars";
			exit(EXIT_FAILURE);
		}
	}
	// ADD NOT EQUALS TO
	// not (x1 = 1 and x2 = 0 and...) = (x1 = 0 or x2 = 1 or ...)
	vector<int> clause;
	for (int i = 0; i < bit_vec.size(); i++) {
		if (values[i] == false)
			clause.push_back(bit_vec[i]);
		else
			clause.push_back(-bit_vec[i]);
	}

	for (int i = 0; i < clause.size(); i++) {
		cout << clause[i] << " ";
	}
	cout << endl;
	cnf.push_back(clause);


	vector<int> clause_bnn2;
	for (int i = 0; i < bit_vec.size(); i++) {
		if (values[i] == false)
			clause_bnn2.push_back(this->input_vars_bnn2[bit_vec[i] - 1]);
		else
			clause_bnn2.push_back(-this->input_vars_bnn2[bit_vec[i] - 1]);
	}

	for (int i = 0; i < clause_bnn2.size(); i++) {
		cout << clause_bnn2[i] << " ";
	}
	cout << endl;
	cnf.push_back(clause_bnn2);


	biggest_aux_var = first_fresh_var;
	write_to_file();
}

void BNNPropertyEncoder::encode_robustness(vector<bool> concrete_input, int perturb, bool equal) {
	cout << "first fresh var after BNNs encoding " << first_fresh_var << endl;
	// add constraints to make x a concrete input
	input_is_equal_to(0, concrete_input);
	cout << "first fresh var after concrete ip encoding " << first_fresh_var << endl;
	if (equal) {
		first_fresh_var = exactly_k_different_inputs(perturb, first_fresh_var);
		cout << "first fresh var after EXACTLY K encoding " << first_fresh_var << endl;
	} else {
		first_fresh_var = at_most_k_different_inputs(perturb, first_fresh_var);
		cout << "first fresh var after AMK encoding " << first_fresh_var << endl;
	}

	first_fresh_var = outputs_are_different(first_fresh_var);
	cout << "first fresh var after outputs are different encoding " << first_fresh_var << endl;

	biggest_aux_var = first_fresh_var;
	write_to_file(1);
}

/*
 * For all possible datasets, what is the change in the output of the function
 * under a 1-bit change.
 */
void BNNPropertyEncoder::encode_dp() {
	int perturb = 1;
	int ret_code;
	first_fresh_var = at_most_k_different_inputs(perturb, first_fresh_var);

	first_fresh_var = outputs_are_different(first_fresh_var);

	biggest_aux_var = first_fresh_var;
	write_to_file();
}


void BNNPropertyEncoder::encode_fairness2(vector<int> locations, vector<bool> values1,
					 vector<bool> values2, vector<vector<int>> dataset_constraints) {
	string orig_out = this->out_filename;
	// old property, without adding constraints over the outputs
	encode_fairness(locations, values1, values2, dataset_constraints);

	for (auto o: this->output_vars_bnn2)
		cout << o << " ";
	cout << endl;
	if (this->output_vars_bnn1.size() != 2 ||
	    this->output_vars_bnn2.size() != 2) {
		cout << "Don't know how to handle multiclass output!" << endl;
		exit(1);
	}

	this->out_filename = orig_out + ".bnn1[0]=1.dimacs";
	this->cnf.push_back({this->output_vars_bnn1[0]});
	write_to_file();

	this->cnf.pop_back();
	this->out_filename = orig_out + ".bnn1[1]=1.dimacs";
	this->cnf.push_back({this->output_vars_bnn1[1]});
	write_to_file();

	this->cnf.pop_back();
	this->out_filename = orig_out + ".bnn2[0]=1.dimacs";
	this->cnf.push_back({this->output_vars_bnn2[0]});
	write_to_file();

	this->cnf.pop_back();
	this->out_filename = orig_out + ".bnn2[1]=1.dimacs";
	this->cnf.push_back(vector<int>{this->output_vars_bnn2[1]});
	write_to_file();
}

// f(x1 x2 ... xn) and xi = values1 and f(x'1... x'n) and x'i = values2 where i in locations and xj = x'j where j not in locations
void BNNPropertyEncoder::encode_fairness(vector<int> locations, vector<bool> values1,
					 vector<bool> values2, vector<vector<int>> dataset_constraints) {
	// fairness constraints are on these specific input variables
	// fix a certain feature to the given value
	add_equals_to(locations, values1, this->cnf);

	// fix a certain feature to the other value
	vector<int> locations2;
	for (int i = 0; i < locations.size(); i++) {
		locations2.push_back(this->input_vars_bnn2[locations[i] - 1]);
	}
	add_equals_to(locations2, values2, this->cnf);

	// all other features are the same, the neural nets differ in just one
	// feature
	for (int i = 0; i < this->input_vars_bnn1.size(); i++) {
		if (find(locations.begin(), locations.end(), this->input_vars_bnn1[i]) == locations.end()) {
			vector<int> clause1;
			clause1.push_back(this->input_vars_bnn1[i]);
			clause1.push_back(-this->input_vars_bnn2[i]);
			this->cnf.push_back(clause1);

			vector<int> clause2;
			clause2.push_back(-this->input_vars_bnn1[i]);
			clause2.push_back(this->input_vars_bnn2[i]);
			this->cnf.push_back(clause2);
		}
	}

	// for how many inputs where only the <locations> inputs differ are we
	// going to get different outputs
	first_fresh_var = outputs_are_different(first_fresh_var);
	cout << "first fresh var after outputs are different encoding " << first_fresh_var << endl;

	// if any, add dataset constraints
	for (auto &ct: dataset_constraints) {
		vector<int> vars;
		for (int i = 0; i < ct.size() - 1; i++) {
			vars.push_back(ct[i]);
		}
		first_fresh_var = add_at_most_bin_k(vars, ct[ct.size() - 1],
						    first_fresh_var, this->cnf);
	}

	biggest_aux_var = first_fresh_var;
	// write both to file
	// projection over the first
	write_to_file();
}

void BNNPropertyEncoder::encode_canary(vector<bool> canary, int non_canary_size) {
	cout << "Encoding canary..." << endl;
	cout << "first fresh var after BNNs encoding " << first_fresh_var << endl;
	cout << "bnn1 input has to be the canary" << endl;
	input_is_equal_to(0, canary);
	cout << "first fresh var after concrete ip encoding " << first_fresh_var << endl;

	// encode in tandem the last output block
	// taking out_var_start and out_var_end of bnn1->blocks[-1] and
	// taking out_var_start and out_var_end of bnn2->blocks[-1] 
	// the output variables are from first_fresh_var

	int out_blk_start = bnn2->blocks[0].in_size + 1;
	cout << "out_blk_start = " << out_blk_start << endl;
	// col_i from bnn2 has to be larger than all - encode the argmax
	// condition for the bnn2
	PBConfig config = make_shared<PBConfigClass>();
	config->amk_encoder = get_enc_type();
	PB2CNF pb2cnf(config);

	float c;
	int bnn2_in_var_start = bnn2->blocks[bnn2->blocks.size() - 1].out_var_start;
	int bnn1_in_var_start = bnn1->blocks[bnn1->blocks.size() - 1].out_var_start;
	cout << "bnn1_in_var_start = " << bnn1_in_var_start << endl;
	cout << "bnn2_in_var_start = " << bnn2_in_var_start << endl;

	// d_ij are the intermediate variables used for the ordering relation
	int d_ij = first_fresh_var;
	cout << "start d_ij = " << d_ij << endl;

	// first fresh var after accounting intermediate variables
	first_fresh_var += bnn2->out_blk.out_size + bnn2->blocks[bnn2->blocks.size() - 1].out_size;
	cout << "first_fresh_var after accounting for intermediate vars = " << first_fresh_var << endl;

	int o = first_fresh_var;
	cout << "output vars start from " << first_fresh_var << endl;
	first_fresh_var += bnn2->out_blk.lin_weight[0].size() + 1;
	cout << "first_fresh_var after accounting for output vars = " << first_fresh_var << endl;

	for (int col_i = 0; col_i < bnn2->out_blk.lin_weight[0].size(); col_i++) {
		vector<WeightedLit> sum_d_ij;

		// constraints for relative output
		vector<WeightedLit> col_bnn1;

		int sum_a_pi = 0;
		int sum_a_pj = 0;

		for (int i = 0; i < bnn1->out_blk.lin_weight.size(); i++) {
			int a_pi = bnn2->out_blk.lin_weight[i][col_i];
			int a_pj = bnn1->out_blk.lin_weight[i][col_i];
			sum_a_pi += a_pi;
			sum_a_pj += a_pj;

			if (a_pi == 1)
				col_bnn1.push_back(WeightedLit(i + bnn2_in_var_start, 1));
			if (a_pi == -1)
				col_bnn1.push_back(WeightedLit(i + bnn2_in_var_start, -1));
			if (a_pj == 1)
				col_bnn1.push_back(WeightedLit(i + bnn1_in_var_start, -1));
			if (a_pj == -1)
				col_bnn1.push_back(WeightedLit(i + bnn1_in_var_start, 1));

		}
		c = bnn1->out_blk.lin_bias[0][col_i] - bnn2->out_blk.lin_bias[0][col_i] + sum_a_pi - sum_a_pj;
		c = ceil(c / 2);
		PBConstraint pbconstraint(col_bnn1, GEQ, c);
		pbconstraint.addConditional(d_ij);

		if (debug) {
			cout << "bnn1 d[" << col_i << "][" << col_i << "]: ";
			pbconstraint.print(false);
		}
		if (debug)
			cout << "before encoding first_fresh_var = " << first_fresh_var << endl;
		VectorClauseDatabase formula(config);
		AuxVarManager auxvars(first_fresh_var);

		pb2cnf.encode(pbconstraint, formula, auxvars);
		first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

		if (debug)
			cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;
		for (auto clause : formula.getClauses())
			this->cnf.push_back(clause);

		PBConstraint pbct(col_bnn1, LEQ, c - 1);
		pbct.addConditional(-d_ij);

		if (debug) {
			cout << "bnn1 d[" << col_i << "][" << col_i << "]: ";
			pbct.print(false);
		}

		if (debug)
			cout << "before encoding first_fresh_var = " << first_fresh_var << endl;
		VectorClauseDatabase f(config);
		AuxVarManager auxv(first_fresh_var);

		pb2cnf.encode(pbct, f, auxv);
		first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;

		for (auto clause : f.getClauses())
			this->cnf.push_back(clause);

		if (debug)
			cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

		sum_d_ij.push_back(WeightedLit(d_ij, 1));
		d_ij += 1;

		for (int col_j = 0; col_j < bnn2->out_blk.lin_weight[0].size(); col_j++) {
			vector<WeightedLit> col;

			if (col_i == col_j)
				continue;

			int sum_a_pi = 0;
			int sum_a_pj = 0;
			int w_minus = 0;
	
			for (int i = 0; i < bnn2->out_blk.lin_weight.size(); i++) {
				int a_pi = bnn2->out_blk.lin_weight[i][col_i];
				int a_pj = bnn2->out_blk.lin_weight[i][col_j];
				sum_a_pi += a_pi;
				sum_a_pj += a_pj;
	
				if (a_pi == 1 && a_pj == -1)
					col.push_back(WeightedLit(i + bnn2_in_var_start, 1));
				else if (a_pi == -1 && a_pj == 1) {
					col.push_back(WeightedLit(-(i + bnn2_in_var_start), 1));
					w_minus++;
				}
			}
			c = bnn2->out_blk.lin_bias[0][col_j] - bnn2->out_blk.lin_bias[0][col_i] + sum_a_pi - sum_a_pj;
			c = ceil(c / 2);
			c = ceil(c / 2) + w_minus;
			PBConstraint pbconstraint(col, GEQ, c);
			pbconstraint.addConditional(d_ij);

			if (debug) {
				cout << "d[" << col_i << "][" << col_j << "]: ";
				pbconstraint.print(false);
			}
			if (debug)
				cout << "before encoding first_fresh_var = " << first_fresh_var << endl;
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
				this->cnf.push_back(clause);

			PBConstraint pbct(col, LEQ, c - 1);
			pbct.addConditional(-d_ij);

			if (debug) {
				cout << "d[" << col_i << "][" << col_j << "]: ";
				pbct.print(false);
			}

			if (debug)
				cout << "before encoding first_fresh_var = " << first_fresh_var << endl;
			VectorClauseDatabase f(config);
			AuxVarManager auxv(first_fresh_var);

			pb2cnf.encode(pbct, f, auxv);
			first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;

			for (auto clause : f.getClauses())
				this->cnf.push_back(clause);

			if (debug)
				cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;
			if (debug == 2)
				for (auto clause : f.getClauses()) {
					for (auto lit : clause) {
						cout << lit << " ";
					}
					cout << "0" << endl;
				}

			sum_d_ij.push_back(WeightedLit(d_ij, 1));
			d_ij += 1;
		}

		PBConstraint pb_out_ct_geq(sum_d_ij, GEQ, bnn2->out_blk.lin_weight[0].size());
		pb_out_ct_geq.addConditional(o);
		if (debug)
			cout << "before encoding first_fresh_var = " << first_fresh_var << endl;

		VectorClauseDatabase formula1(config);
		AuxVarManager auxvars1(first_fresh_var);

		pb2cnf.encode(pb_out_ct_geq, formula1, auxvars1);
		first_fresh_var = auxvars1.getBiggestReturnedAuxVar() + 1;

		for (auto clause : formula1.getClauses())
			this->cnf.push_back(clause);

		if (debug == 2) {
			pb_out_ct_geq.print(false);
			for (auto clause : formula1.getClauses()) {
				for (auto lit : clause) {
					cout << lit << " ";
				}
				cout << "0" << endl;
			}
		}

		PBConstraint pb_out_ct_leq(sum_d_ij, LEQ, bnn2->out_blk.lin_weight[0].size());
		pb_out_ct_leq.addConditional(o);
		if (debug)
			cout << "before encoding first_fresh_var = " << first_fresh_var << endl;
		VectorClauseDatabase formula2(config);
		AuxVarManager auxvars2(first_fresh_var);

		pb2cnf.encode(pb_out_ct_leq, formula2, auxvars2);
		first_fresh_var = auxvars2.getBiggestReturnedAuxVar() + 1;

		for (auto clause : formula2.getClauses())
			this->cnf.push_back(clause);

		if (debug) {
			pb_out_ct_leq.print(false);
			for (auto clause : formula2.getClauses()) {
				for (auto lit : clause) {
					cout << lit << " ";
				}
				cout << "0" << endl;
			}
		}

		PBConstraint pbo2(sum_d_ij, LEQ, bnn2->out_blk.lin_weight[0].size() - 1);
		pbo2.addConditional(-o);
		if (debug)
			cout << "before encoding first_fresh_var = " << first_fresh_var << endl;


		VectorClauseDatabase f2(config);
		AuxVarManager auxv2(first_fresh_var);

		pb2cnf.encode(pbo2, f2, auxv2);
		first_fresh_var = auxv2.getBiggestReturnedAuxVar() + 1;

		for (auto clause : f2.getClauses())
			this->cnf.push_back(clause);

		if (debug == 2) {
			pbo2.print(false);
			cout << "CNF ^" << endl;
			for (auto clause : f2.getClauses()) {
				for (auto lit : clause) {
					cout << lit << " ";
				}
				cout << "0" << endl;
			}
		}
		o += 1;
	}

       /* cout << "fix first " << non_canary_size << "inputs of bnn2" << endl;*/
	//vector<int> fixed_non_canary;
	//vector<bool> fixed_non_canary_values;

	//for (int i = 0; i < non_canary_size; i++) {
		//fixed_non_canary.push_back(this->input_vars_bnn2[i]);
		//fixed_non_canary_values.push_back(true);
		//cout << "fixing bnn2's " << fixed_non_canary[i] << " = " << fixed_non_canary_values[i] << endl;
	//}

	//add_equals_to(fixed_non_canary, fixed_non_canary_values, this->cnf);

	biggest_aux_var = first_fresh_var;
	write_to_file(1);
}

// FIXME add some checks or ENUM for which_proj_bnn
// which_proj_bnn writes the projection variables either the input variables of
// bnn1 or the input variables of bnn2
// which_proj_bnn = 0 if projection on the first network's input
// which_proj_bnn = 1 if projection on the second network's input
// which_bnn = 0 if write to file only the first bnn
// which_bnn = 1 if write to file only the second bnn
// which_bnn = 2 if write to file both bnns
void BNNPropertyEncoder::write_to_file(int which_proj_bnn, int which_bnn) {
	int total_clauses = cnf.size();
	ofstream out_file;

	out_file.open(this->out_filename, ios::out);

	if (out_file.is_open()) {

		for (int i = 0; i < bnn1->num_internal_blocks; i++) {
			total_clauses += bnn1->blocks[i].cnf_formula.size();
		}

		for (int i = 0; i < bnn2->num_internal_blocks; i++) {
			total_clauses += bnn2->blocks[i].cnf_formula.size();
		}

		total_clauses += bnn1->out_blk.cnf_formula.size();
		total_clauses += bnn2->out_blk.cnf_formula.size();
		out_file << "p cnf " << this->biggest_aux_var - 1 << " " << total_clauses << endl;

		/*for (int i = 1; i <= bnn1->blocks[0].in_size; i++) {*/
			//string str = "c ind ";
			//int j = 0;
			//for (; j < 9; j++) {
				//if (i + j <= bnn1->blocks[0].in_size)
					//str += to_string(i + j) + " ";
				//else {
					//str += " ";
					//break;
				//}
			//}
			//i += j - 1;
			//out_file << str << "0" << endl;
		/*}*/
		for (int i = 0; i < input_vars_bnn1.size(); i++) {
			string str = "c ind ";
			int j = 0;
			for (; j < 9; j++) {
				if (i + j < input_vars_bnn1.size())
					if (which_proj_bnn == 0)
						str += to_string(input_vars_bnn1[i + j]) + " ";
					else
						str += to_string(input_vars_bnn2[i + j]) + " ";
				else {
					str += " ";
					break;
				}
			}
			i += j - 1;
			out_file << str << "0" << endl;
		}


		if (which_bnn == 0 || which_bnn == 2)
			bnn1->write_to_file(out_file);

		if (which_bnn >= 1)
			bnn2->write_to_file(out_file);

		/* property encoding */
		for (auto clause : cnf) {
			for (auto lit : clause)
				out_file << lit << " ";
			out_file << "0" << endl;
		}

		/* write any native xor constraints */
		/* THIS IS CRYPTOMINISAT-specific */
		for (auto clause: xors) {
			out_file << "x ";
			for (auto lit: clause)
				out_file << lit << " ";
			out_file << "0" << endl;
		}

		out_file.close();
		cout << "Finished writing CNF to " << out_filename << endl;
	} else {
		cout << "Error opening output file " << out_filename << endl;
		throw std::runtime_error("Could not open file " + out_filename);
	}
}
