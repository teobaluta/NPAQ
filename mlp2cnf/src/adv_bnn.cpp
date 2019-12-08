#include "adv_bnn.h"

AdvBNN::AdvBNN(string model_dir, string out_dir, string perturb_filename) {
	ifstream perturb_file;

	perturb_file.open(perturb_filename);
	if (perturb_file.is_open()) {
		int var_flipped;

		while (perturb_file >> var_flipped) {
			perturb.insert(var_flipped);
		}

	} else {
		cout << "Error opening output file " << perturb_filename << endl;
		throw std::runtime_error("Error opening output file " + perturb_filename);
	}

	this->model_dir = model_dir;
	this->out_dir = out_dir;
	cout << "Creating a BNN..." << endl;
	bnn = new BNNModel(model_dir, out_dir);

	if (debug) {
		set<int>::iterator it;
		cout << "flipping: ";
		for (it = perturb.begin(); it != perturb.end(); ++it) {
			cout << *it << " ";
		}
		cout << endl;
	}

	cout << "Creating a perturb BNN..." << endl;
	perturb_bnn = new BNNModel(model_dir, out_dir, perturb);
}

int AdvBNN::encode() {
	ofstream out_file;
	string out_filename;
	int first_fresh_var;
	int biggest_aux_var;
	char separator;
#ifdef _WIN32
	separator = '\\';
#else
	separator = '/';
#endif

	out_filename = this->out_dir + separator + "adv_" + to_string(this->perturb.size()) + ".dimacs";

	cout << "Writing formula to " << out_filename << endl;
	int out_var_start = bnn->blocks[0].in_size;

	for (int i = 0; i < bnn->num_internal_blocks; i++) {
		cout << "blk " << i << " size " << bnn->blocks[i].out_size << endl;
		out_var_start += bnn->blocks[i].out_size + bnn->out_blk.lin_weight[0].size() + 1;
	}

	out_var_start += bnn->out_blk.out_size;
	// this is actually the number of intermediate variables in the
	// cardinality constraints
	cout << "BNN out_var_end = " << out_var_start << endl;

	// first fresh var starts after reserving variables for intermediate
	// layers, since the two BNNs are the same architecture just double the
	// number of reserved vars, but subtract the input vars (we only count
	// those once)
	biggest_aux_var = bnn->encode(2*out_var_start - bnn->blocks[0].in_size - 1);
	first_fresh_var = bnn->out_blk.lin_weight[0].size() + out_var_start + biggest_aux_var;
	cout << "perturb first_fresh_var = " << first_fresh_var << endl;

	biggest_aux_var = perturb_bnn->encode(first_fresh_var, 1, out_var_start) + 1;

	vector<vector<int>> ineq_clauses;
	int total_clauses = 0;
	// do the O1 =! O2 encoding
	int bnn1_out_start = bnn->blocks[0].in_size + 1;
	int bnn2_out_start = out_var_start;
	// will introduce a new variable for each O1 =! O1'
	for (int i = 0; i < bnn->out_blk.lin_weight[0].size(); i++) {
		cout << "Encoding " << bnn1_out_start << " != " << bnn2_out_start << endl;
		// this is basically tseitin
		// introduce p <-> (a!=b)
		// p -> (a!=b) = (not p or a or b) and (not p or not a or not b)
		// p <- (a!=b) = (p or not a or b) and (p or a or not b)
		vector<int> clause1;
		clause1.push_back(-biggest_aux_var);
		clause1.push_back(bnn1_out_start);
		clause1.push_back(bnn2_out_start);
		ineq_clauses.push_back(clause1);

		vector<int> clause2;
		clause2.push_back(-biggest_aux_var);
		clause2.push_back(-bnn1_out_start);
		clause2.push_back(-bnn2_out_start);
		ineq_clauses.push_back(clause2);

		vector<int> clause3;
		clause3.push_back(biggest_aux_var);
		clause3.push_back(-bnn1_out_start);
		clause3.push_back(bnn2_out_start);
		ineq_clauses.push_back(clause3);

		vector<int> clause4;
		clause4.push_back(biggest_aux_var);
		clause4.push_back(bnn1_out_start);
		clause4.push_back(-bnn2_out_start);
		ineq_clauses.push_back(clause4);

		bnn1_out_start++;
		bnn2_out_start++;
		biggest_aux_var++;
	}

	int go_back_var = biggest_aux_var - 1;
	vector<int> last_clause;
	for (int i = 0; i < bnn->out_blk.lin_weight[0].size(); i++) {
		last_clause.push_back(go_back_var);
		go_back_var--;
	}
	ineq_clauses.push_back(last_clause);
	total_clauses += ineq_clauses.size();

	// XXX should really refactor this into a write_encoding_dimacs() method of the BNN
	out_file.open(out_filename);
	if (out_file.is_open()) {
		cout << "Writing encoding to " << out_filename << endl;

		for (int i = 0; i < bnn->num_internal_blocks; i++) {
			total_clauses += bnn->blocks[i].cnf_formula.size() + perturb_bnn->blocks[i].cnf_formula.size();
		}

		total_clauses += bnn->out_blk.cnf_formula.size();
		total_clauses += perturb_bnn->out_blk.cnf_formula.size();

		out_file << "p cnf " << biggest_aux_var - 1 << " " << total_clauses << endl;

		for (int i = 0; i < bnn->num_internal_blocks; i++)
			for (auto clause : bnn->blocks[i].cnf_formula) {
				for (auto lit : clause)
					out_file << lit << " ";
				out_file << "0" << endl;
			}

		for (auto clause : bnn->out_blk.cnf_formula) {
			for (auto lit : clause)
				out_file << lit << " ";
			out_file << "0" << endl;
		}

		for (int i = 0; i < perturb_bnn->num_internal_blocks; i++)
			for (auto clause : perturb_bnn->blocks[i].cnf_formula) {
				for (auto lit : clause)
					out_file << lit << " ";
				out_file << "0" << endl;
			}

		for (auto clause : perturb_bnn->out_blk.cnf_formula) {
			for (auto lit : clause)
				out_file << lit << " ";
			out_file << "0" << endl;
		}

		for (auto clause : ineq_clauses) {
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
