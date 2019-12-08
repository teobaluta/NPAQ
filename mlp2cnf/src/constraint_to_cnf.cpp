#include "constraint_to_cnf.h"


/*
 * we can directly encode the xor constraint in cryptominisat
 */
int add_xor_directly(vector<int> bit_vec1, vector<int> bit_vec2,
		     vector<vector<int>> &xors, vector<int> &aux_vars,
		     int first_fresh_var)
{
	if (bit_vec1.size() != bit_vec2.size()) {
		cout << "Bitvectors are of different sizes; cannot xor" << endl;
		throw std::runtime_error("Bitvectors are of different sizes; cannot xor");
	}

	if (first_fresh_var < 1) {
		cout << "Cannot start variables with " << first_fresh_var << endl;
		throw std::runtime_error("Cannot start variables with " + first_fresh_var);
	}

	for (int i = 0; i < bit_vec1.size(); i++) {
		vector<int> xor_clause;
		xor_clause.push_back(-first_fresh_var);
		xor_clause.push_back(bit_vec1[i]);
		xor_clause.push_back(bit_vec2[i]);
		xors.push_back(xor_clause);
		aux_vars.push_back(first_fresh_var);
		first_fresh_var++;
	}

	return first_fresh_var;
}

int add_xor_constraint(vector<int> bit_vec1, vector<int> bit_vec2,
		       vector<vector<int>> &cnf, vector<int> &aux_vars,
		       int first_fresh_var) {
	if (bit_vec1.size() != bit_vec2.size()) {
		cout << "Bitvectors are of different sizes; cannot xor" << endl;
		throw std::runtime_error("Bitvectors are of different sizes; cannot xor");
	}

	if (first_fresh_var < 1) {
		cout << "Cannot start variables with " << first_fresh_var << endl;
		throw std::runtime_error("Cannot start variables with " + first_fresh_var);
	}

	for (int i = 0; i < bit_vec1.size(); i++) {
		// this is basically tseitin
		// introduce p <-> (a!=b)
		// p -> (a!=b) = (not p or a or b) and (not p or not a or not b)
		// p <- (a!=b) = (p or not a or b) and (p or a or not b)
		vector<int> clause1;
		clause1.push_back(-first_fresh_var);
		clause1.push_back(bit_vec1[i]);
		clause1.push_back(bit_vec2[i]);
		cnf.push_back(clause1);

		vector<int> clause2;
		clause2.push_back(-first_fresh_var);
		clause2.push_back(-bit_vec1[i]);
		clause2.push_back(-bit_vec2[i]);
		cnf.push_back(clause2);

		vector<int> clause3;
		clause3.push_back(first_fresh_var);
		clause3.push_back(-bit_vec1[i]);
		clause3.push_back(bit_vec2[i]);
		cnf.push_back(clause3);

		vector<int> clause4;
		clause4.push_back(first_fresh_var);
		clause4.push_back(bit_vec1[i]);
		clause4.push_back(-bit_vec2[i]);
		cnf.push_back(clause4);

		aux_vars.push_back(first_fresh_var);
		first_fresh_var++;
	}

	return first_fresh_var;
}

/*
 * PBLib encode at-most-k constraint, i.e. sum over the bits of bit_vec is less
 * or equal to k
 */
int add_at_most_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf) {
	vector<WeightedLit> literals;
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);

	for (int i = 0; i < bit_vec.size(); i++)
		literals.push_back(WeightedLit(bit_vec[i], 1));

	PBConstraint pbconstraint(literals, LEQ, k);

	VectorClauseDatabase formula(config);
	AuxVarManager auxvars(first_fresh_var);

	pb2cnf.encode(pbconstraint, formula, auxvars);
	first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	if (debug)
		pbconstraint.print(false);
	for (auto clause : formula.getClauses()) {
		cnf.push_back(clause);
	}

	return first_fresh_var;
}

/* Something for binary encoding for dataset */
int add_at_most_bin_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf) {
	vector<WeightedLit> literals;
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);

	for (int i = bit_vec.size() - 1; i >= 0; i--)
		literals.push_back(WeightedLit(bit_vec[i], pow(2, i)));

	PBConstraint pbconstraint(literals, LEQ, k);

	VectorClauseDatabase formula(config);
	AuxVarManager auxvars(first_fresh_var);

	pb2cnf.encode(pbconstraint, formula, auxvars);
	first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	pbconstraint.print(false);
	for (auto clause : formula.getClauses()) {
		cnf.push_back(clause);
	}

	return first_fresh_var;
}
int add_at_least_k(vector<int> bit_vec, int k, int first_fresh_var, vector<vector<int>> &cnf) {
	vector<WeightedLit> literals;
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);

	for (int i = 0; i < bit_vec.size(); i++)
		literals.push_back(WeightedLit(bit_vec[i], 1));

	PBConstraint pbconstraint(literals, GEQ, k);

	VectorClauseDatabase formula(config);
	AuxVarManager auxvars(first_fresh_var);

	pb2cnf.encode(pbconstraint, formula, auxvars);
	first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	if (debug)
		pbconstraint.print(false);
	for (auto clause : formula.getClauses()) {
		cnf.push_back(clause);
	}

	return first_fresh_var;
}

/*
 * For every bit i in the bit vector, add constraint to be either 0 or 1
 * depending on the values[i].
 */
int add_equals_to(vector<int> bit_vec, vector<bool> values, vector<vector<int>> &cnf) {
	if (values.size() != bit_vec.size()) {
		cout << "Symbolic bit vector size " << bit_vec.size() << " is different from values vector size " << values.size() << "! Quiting..." << endl;
		throw std::runtime_error("Symbolic bit vector size is different from values vector size! Quiting...");
	}

	for (int i = 0; i < bit_vec.size(); i++) {
		vector<int> clause;
		if (values[i] == false)
			clause.push_back(-bit_vec[i]);
		else if (values[i] == true)
			clause.push_back(bit_vec[i]);
		else {
			cout << "The values vector should be a bit vector, values are only 0 or 1.";
			throw std::runtime_error("The values vector should be a bit vector, values are only 0 or 1.");
		}
		cnf.push_back(clause);
	}
}

/*
 * Set the i-th bit in the bit_vec to a value
 */
void add_equals_to(vector<int> bit_vec, int idx, bool value, vector<vector<int>> &cnf) {
	vector<int> clause;
	if (value == true)
		clause.push_back(bit_vec[idx]);
	else if (value == false)
		clause.push_back(-bit_vec[idx]);

	cnf.push_back(clause);
}

int add_equals_to(vector<int> bit_vec1, vector<int> bit_vec2, vector<vector<int>> &cnf) {
	if (bit_vec1.size() != bit_vec2.size()) {
		cout << "Symbolic bit vector size is different from values vector size! Quiting..." << endl;
		throw std::runtime_error("Symbolic bit vector size is different from values vector size! Quiting...");
	}

	for (int i = 0; i < bit_vec1.size(); i++) {
		vector<int> clause1;
		clause1.push_back(bit_vec1[i]);
		clause1.push_back(-bit_vec2[i]);
		cnf.push_back(clause1);

		vector<int> clause2;
		clause2.push_back(-bit_vec1[i]);
		clause2.push_back(bit_vec2[i]);
		cnf.push_back(clause2);
	}
}
