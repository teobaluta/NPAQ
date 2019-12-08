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

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "./example output_file" << endl;
	}

	vector<WeightedLit> col;
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);

	int i = 1;
	for (i = 1; i < 6; i++) {
		col.push_back(WeightedLit(i, 2*i / 3 + 1));
	}

	PBConstraint pbct(col, LEQ, 3);
	pbct.addConditional(6);

	pbct.print(false);

	VectorClauseDatabase formula(config);
	AuxVarManager auxvars(7);

	pb2cnf.encode(pbct, formula, auxvars);
	int first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	PBConstraint pbct_neg(col, GEQ, 4);
	pbct_neg.addConditional(-6);

	pbct_neg.print(false);

	VectorClauseDatabase formula_neg(config);
	AuxVarManager auxvars_neg(first_fresh_var);

	pb2cnf.encode(pbct_neg, formula_neg, auxvars_neg);
	first_fresh_var = auxvars_neg.getBiggestReturnedAuxVar() + 1;

	ofstream out_file;
	out_file.open(argv[1]);
	out_file << "p cnf " << auxvars_neg.getBiggestReturnedAuxVar() << " " << formula.getClauses().size() + formula_neg.getClauses().size() << endl;
	for (auto clause : formula.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "0" << endl;
	}

	for (auto clause : formula_neg.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "0" << endl;
	}

	return 0;
}
