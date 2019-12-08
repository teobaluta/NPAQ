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
	AuxVarManager auxvars(8);

	pb2cnf.encode(pbct, formula, auxvars);
	int first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	col.clear();
	for (int i = 1; i < 6; i++) {
		col.push_back(WeightedLit(i, 3*i / 2 + 1));
	}

	PBConstraint pbct2(col, GEQ, 6);
	pbct2.addConditional(7);

	pbct2.print(false);

	VectorClauseDatabase formula2(config);
	AuxVarManager auxvars2(first_fresh_var);

	pb2cnf.encode(pbct2, formula2, auxvars2);
	first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

	ofstream out_file;
	out_file.open(argv[1]);
	out_file << "p cnf " << auxvars2.getBiggestReturnedAuxVar() << " " << 2 * formula.getClauses().size() + 2 * formula2.getClauses().size() << endl;

	for (auto clause : formula.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "0" << endl;
	}

	for (auto clause : formula.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "-6 0" << endl;
	}


	for (auto clause : formula2.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "0" << endl;
	}

	for (auto clause : formula2.getClauses()) {
		for (auto lit : clause) {
			out_file << lit << " ";
		}
		out_file << "-7 0" << endl;
	}



	return 0;
}
