#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "pb2cnf.h"
#include "utils.h"

using namespace std;
using namespace PBLib;

string OUTPUT_FILE = "cnf.dimacs";
string OUT_DIR = "out";
string META_FILE = "meta.txt";

const int CSVERR_W = 10;
const int CSVERR_B = 11;

int debug = 0;

inline string filename(string path, int number, bool weights_file = true) {
	char separator;
#ifdef _WIN32
	separator = '\\';
#else
	separator = '/';
#endif

	if (weights_file == true)
		return path + separator + "weights_l" + to_string(number) + ".csv";
	return path + separator + "bias_l" + to_string(number) + ".csv";
}


/* XXX OBSOLETE; no need to do manual Tseitin transformation
 * Transforms the constraint -> output_var
 * e.g. input: -7 x1 +5 ~x2 +9 ~x3 -3 ~x4 +7 x4 >= 0 -> x5 = 1
 * output:
 *
 * p cnf 17 26
 * -10 -5 0
 * ...
 * 5 10 11 12 13 14 15 16 17 0
 */
int constraint2cnf(PBConstraint pbconstraint, int first_fresh_var, int output_var,
		   vector<vector<int>> *cnf_formula) {
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);
	VectorClauseDatabase formula(config);
	AuxVarManager auxvars(first_fresh_var);

	pb2cnf.encode(pbconstraint, formula, auxvars);

	if (debug) {
		pbconstraint.print(false);
		ofstream orig_file;
		stringstream debug_sstm;
		debug_sstm << OUT_DIR << "/ct_" << first_fresh_var;
		string debug_orig_ct = debug_sstm.str();

		cout << "Writing to " << debug_orig_ct << endl;
		orig_file.open(debug_orig_ct);
		orig_file << "p cnf " << auxvars.getBiggestReturnedAuxVar() << " "
			  << formula.getClauses().size() << endl;

		for (auto clause : formula.getClauses()) {
			for (auto lit : clause) {
				orig_file << lit << " ";
			}
			orig_file << "0" << endl;
		}
	}

	// We have obtained formula in CNF, let's do f implies v
	int alfa_var = auxvars.getBiggestReturnedAuxVar() + 1;
	cout << "Encoded PB. Adding implication with alfa >= " << alfa_var << endl;

	// last term of resulting CNF is (output_var or alfa_var_i)
	vector<int> clause_output_var;
	clause_output_var.push_back(output_var);

	/*
	 * Use Tseitin Transformation to add implication (F -> output_var)
	 * F -> output_var equisatisf with not F or output_var
	 *	- not F = not C1 or not C2 or not C3 ...
	 *	- add an extra variable alfa_i <-> not C_i
	 *	- (not alfa_i or not l1_i) and (not alfa_i or not l2_i) and ... and
	 *	(l1_i or l2_i or .. or alfa_i)
	 */
	for (auto clause : formula.getClauses()) {
		vector<int> or_lit_or_alfa;
		for (auto lit : clause) {
			vector<int> terms_1 = {-alfa_var, -lit};
			(*cnf_formula).push_back(terms_1);
			or_lit_or_alfa.push_back(lit);
		}
		or_lit_or_alfa.push_back(alfa_var);
		(*cnf_formula).push_back(or_lit_or_alfa);

		// F or not output_var (F <- output_var)
		clause.push_back(-output_var);
		(*cnf_formula).push_back(clause);

		clause_output_var.push_back(alfa_var);
		alfa_var++;
	}

	(*cnf_formula).push_back(clause_output_var);

	if (debug) {
		ofstream out_file;
		stringstream debug_cnf_sstm;
		debug_cnf_sstm << OUT_DIR << "/cnf_ct_" << first_fresh_var;
		string debug_orig_cnf_ct = debug_cnf_sstm.str();

		out_file.open(debug_orig_cnf_ct);
		cout << "Writing to " << debug_orig_cnf_ct << endl;
		out_file << "p cnf " << alfa_var - 1 << " " << (*cnf_formula).size() << endl;

		for (int i = 0; i < (*cnf_formula).size(); i++) {
			for (auto lit : (*cnf_formula)[i]) {
				out_file << lit << " ";
			}
			out_file << "0" << endl;
		}
	}

	cout << "Encoded PB -> " << output_var <<"; alfa < " << alfa_var << endl;
	return alfa_var;
}

class LinearLayer {
      public:
	int in_var_start;
	int in_var_end;
	vector<vector<WeightedLit>> constraints;
	int no_constraints;
	vector<vector<float>> biases;
	vector<vector<int>> new_formula_cnf;
	int output_var;
	int output_var_end;

	LinearLayer(string weights_csv, string biases_csv, int var_start = 1,
		    int output_var_start = 0) {
		vector<vector<float>> weights;

		// XXX should check the size, what if parsing fails
		weights = parseCSV(weights_csv);
		if (weights.size() == 0) {
			throw CSVERR_W;
		}
		this->biases = parseCSV(biases_csv);
		if (this->biases.size() == 0) {
			throw CSVERR_B;
		}
		this->in_var_start = var_start;
		this->no_constraints = weights[0].size();

		this->in_var_end = this->in_var_start + weights.size() - 1;
		if (output_var_start == 0)
			this->output_var = this->in_var_end + 1;
		else
			this->output_var = output_var_start;

		this->output_var_end = this->output_var + weights[0].size();

		//cout << "weights size " << weights.size() << endl;
		//cout << "weights[0] size " << weights[0].size() << endl;

		for (int j = 0; j < weights[0].size(); j++) {
			vector<WeightedLit> col_weights;
			for (int i = 0; i < weights.size(); i++) {
				if (round(weights[i][j]) == 0)
					continue;
				col_weights.push_back(
				    WeightedLit(i + in_var_start, round(weights[i][j])));
			}
			this->constraints.push_back(col_weights);
		}

	}

	// XXX destructor


	// XXX: OBSOLETE; have to delete this
	int encode_obs(int first_fresh_var) {
		PBConfig config = make_shared<PBConfigClass>();
		PB2CNF pb2cnf(config);

		if (this->constraints.size() != biases.size()) {
			cout << "Literals size (" << this->constraints.size()
			     << ") and biases size (" << biases.size() << ") differ!" << endl;
			return -1;
		}
		int output_var;

		for (int i = 0; i < this->constraints.size(); i++) {
			// hardcoded for sigmoid 0.5
			// w * x + b >= 0.5 to activate neuron
			PBConstraint pbconstraint(this->constraints[i], GEQ,
						  round(0.5 - biases[i][0]));

			output_var = this->output_var + i;
			first_fresh_var = constraint2cnf(pbconstraint, first_fresh_var, output_var,
							 &this->new_formula_cnf);

		}

		cout << "Layer out_var [" << this->output_var << " - "
			<< this->output_var_end << "]" << endl;
		return first_fresh_var;
	}

	int encode_layer(int first_fresh_var, float thresh) {
		PBConfig config = make_shared<PBConfigClass>();
		PB2CNF pb2cnf(config);

		if (this->constraints.size() != biases.size()) {
			cout << "Literals size (" << this->constraints.size()
			     << ") and biases size (" << biases.size() << ") differ!" << endl;
			return -1;
		}

		int output_var;

		//cout << "Encode layer with first_fresh_var from " << first_fresh_var << endl;
		for (int i = 0; i < this->constraints.size(); i++) {
			// hardcoded for sigmoid 0.5
			// w * x + b >= 0.5 to activate neuron
			PBConstraint pbconstraint(this->constraints[i], GEQ,
						  round(thresh - biases[i][0]));

			//output_var = this->in_var_end + i + 1;
			output_var = this->output_var + i;

			pbconstraint.addConditional(output_var);

			if (debug)
				pbconstraint.print(false);
			VectorClauseDatabase formula(config);
			AuxVarManager auxvars(first_fresh_var);

			pb2cnf.encode(pbconstraint, formula, auxvars);
			first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;
			if (debug)
				cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

			if (debug == 2) {
				for (auto clause : formula.getClauses()) {
					for (auto lit : clause) {
						cout << lit << " ";
					}
					cout << "0" << endl;
				}
			}

			for (auto clause : formula.getClauses()) {
				this->new_formula_cnf.push_back(clause);
			}

			int rhs = round(thresh - biases[i][0]) - 1;

			PBConstraint pbconstraint_neg(this->constraints[i], LEQ, rhs);
			pbconstraint_neg.addConditional(-output_var);

			if (debug)
				pbconstraint_neg.print(false);

			// XXX can just use the same formula and auxvars
			VectorClauseDatabase formula_neg(config);
			AuxVarManager auxvars_neg(first_fresh_var);
			pb2cnf.encode(pbconstraint_neg, formula_neg, auxvars_neg);
			first_fresh_var = auxvars_neg.getBiggestReturnedAuxVar() + 1;

			if (debug)
				cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

			if (debug == 2) {
				for (auto clause : formula_neg.getClauses()) {
					for (auto lit : clause) {
						cout << lit << " ";
					}
					cout << "0" << endl;
				}
			}

			for (auto clause : formula_neg.getClauses()) {
				this->new_formula_cnf.push_back(clause);
			}
		}

		if (debug) {
			cout << "Layer out_var [" << this->output_var << " - " << this->output_var_end
				<< "]" << endl;
			cout << "Return " << first_fresh_var << endl;
		}
		return first_fresh_var;
	}
};

class LinearModel {
      public:
	vector<LinearLayer> layers;
	vector<vector<int>> cnf_formula;
	int reserved_var = 0;
	int no_layers;
	string out_dir;
	int input_vars[2];
	int output_vars[2];

	LinearModel(string model_dir, string out_dir, int no_layers = 2) {
		// model_dir should contain the weights and biases for each layer
		this->no_layers = no_layers;
		// first variable is x1
		int var_start = 1;
		this->out_dir = out_dir;

		input_vars[0] = var_start;

		// reserve variables for input and output first
		string weights_csv;
		string biases_csv;
		vector<vector<float>> weights;

		// input layer
		weights_csv = filename(model_dir, 1);
		biases_csv = filename(model_dir, 1, false);
		weights = parseCSV(weights_csv);
		if (weights.size() == 0) {
			throw CSVERR_W;
		}
		var_start += weights.size();

		// output layer
		// read output size to reserve for output variables for it
		weights_csv = filename(model_dir, no_layers);

		// XXX should check the size, what if parsing fails
		weights = parseCSV(weights_csv);
		if (weights.size() == 0) {
			throw CSVERR_W;
		}
		var_start += weights[0].size();

		// reserve variables for each constraint variables
		// in the inner layers
		for (int i = 0; i < no_layers; i++) {
			weights_csv = filename(model_dir, i + 1);
			biases_csv = filename(model_dir, i + 1, false);

			if (i == no_layers - 1)
				layers.push_back(LinearLayer(weights_csv, biases_csv, var_start,
							     layers[0].in_var_end + 1));
			else if (i == 0)
				layers.push_back(LinearLayer(weights_csv, biases_csv, 1, var_start));
			else
				layers.push_back(LinearLayer(weights_csv, biases_csv, var_start));

			if (debug) {
				cout << "Layer " << i << ": #constraint= " << layers[i].no_constraints
					<< endl;
				cout << "Layer " << i << " in_var [" << layers[i].in_var_start << " - "
					<< layers[i].in_var_end << "]" << endl;
			}
			var_start = layers[i].output_var;
		}
		this->reserved_var +=
		    layers[no_layers - 1].in_var_end + layers[no_layers - 1].no_constraints;

		input_vars[0] = layers[0].in_var_start;
		input_vars[1] = layers[0].in_var_end;

		output_vars[0] = layers[no_layers - 1].output_var;
		output_vars[1] = layers[no_layers - 1].output_var_end;

		ofstream meta_fs;

		meta_fs.open(out_dir + "/" + META_FILE);
		if (meta_fs.is_open()) {

			meta_fs << input_vars[0] << " " << input_vars[1] << endl;
			meta_fs << output_vars[0] << " " << output_vars[1] << endl;
		} else {
			cout << "Error opening meta file." << endl;
		}
		meta_fs.close();

		if (debug) {
			cout << "Reserved variables " << this->reserved_var << endl;
		}
	}

	// XXX destructor

	int encode_model(float thresh) {
		int biggest_aux_var;
		int first_fresh_var = this->reserved_var + 1;
		ofstream out_file;

		if (layers.size() == 0)
			return 1;

		for (int i = 0; i < no_layers; i++) {
			if (debug) {
				cout << "Encoding Layer " << i << endl;
			}
			biggest_aux_var = layers[i].encode_layer(first_fresh_var, thresh);
			if (biggest_aux_var < 0) {
				cout << "Error encoding layer " << i << ": overflow auxvar" << endl;
				return 1;
			}

			for (auto clause : layers[i].new_formula_cnf) {
				cnf_formula.push_back(clause);
			}

			first_fresh_var = biggest_aux_var;
		}

		string out_filename(this->out_dir + "/" + OUTPUT_FILE);
		out_file.open(out_filename);
		if (out_file.is_open()) {
			cout << "Writing encoding to " << out_filename << endl;

			out_file << "p cnf " << biggest_aux_var << " " << cnf_formula.size() << endl;

			for (auto clause : cnf_formula) {
				for (auto lit : clause)
					out_file << lit << " ";
				out_file << "0" << endl;
			}
		} else {
			cout << "Error opening output file " << out_filename << endl;
			return 1;
		}

		return 0;
	}
};


int main(int argc, char *argv[]) {
	if (argc > 5 || argc < 4) {
		cout << "./mlp2cnf $model_dir $output_dir sigmoid|relu $debug" << endl;
		return 1;
	}

	int err_dir;
	int ret_code = 0;
	float thresh = 0.5;

	err_dir = exists_dir(argv[1]);
	if (err_dir) {
		return err_dir;
	}

	err_dir = ensure_dir(argv[2]);
	if (err_dir) {
		return err_dir;
	}

	string model_dir(argv[1]);
	string out_dir(argv[2]);
	string activation(argv[3]);

	if (activation.compare("sigmoid") == 0)
		thresh = 0.5;
	else if (activation.compare("relu") == 0)
		thresh = 0.0;
	else {
		cout << "Error: activation can be 'sigmoid' or 'relu'" << endl;
		return 1;
	}

	if (argc == 5) {
		debug = atoi(argv[4]);
		if (debug != 0 && debug != 1 && debug != 2) {
			cout << "Available debug options are 0 (no debug), 1" <<
				" (minimal) and 2 (verbose)" << endl;
			return 1;
		}
	}

	OUT_DIR = out_dir;
	try {
		LinearModel model(model_dir, out_dir);
		ret_code = model.encode_model(thresh);
	} catch (int e) {
		switch (e) {
			case CSVERR_W:
				cout << "Error reading model's weights csv." << endl;
				break;
			case CSVERR_B:
				cout << "Error reading model's biases csv." << endl;
				break;
			default:
				cout << "Error creating LinearModel." << endl;
		}
	}

	return ret_code;
}
