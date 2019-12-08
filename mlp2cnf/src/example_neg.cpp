#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <sstream>

#include "pb2cnf.h"
#include "utils.h"
#include "dirent.h"
#include "z3++.h"

using namespace z3;

using namespace std;
using namespace PBLib;

context z3ctx;

void visit(expr const & e) {
	if (e.is_app()) {
		unsigned num = e.num_args();
		for (unsigned i = 0; i < num; i++) {
			visit(e.arg(i));
		}
		// do something
		// Example: print the visited expression
		func_decl f = e.decl();
		std::cout << "application of " << f.name() << ": " << e << "\n";
	}
	else if (e.is_quantifier()) {
		visit(e.body());
		// do something
	}
	else { 
		assert(e.is_var());
		// do something
	}
}

void print_dimacs(expr const & e, int *num_clauses, map<string, string> *vars, ostream& out, int *start_var) {
	if (e.is_and())
		for (unsigned int i = 0; i < e.num_args(); i++) {
			*num_clauses = *num_clauses + 1;
			print_dimacs(e.arg(i), num_clauses, vars, out, start_var);
			out << "0" << endl;
		}
	else if (e.is_or())
		for (unsigned int i = 0; i < e.num_args(); i++) {
			print_dimacs(e.arg(i), num_clauses, vars, out, start_var);
		}
	else if (e.is_not()) {
		func_decl f = e.arg(0).decl();
		string out_name;
		if (f.name().kind() == Z3_INT_SYMBOL) {
			out_name = to_string(*start_var);
			auto s = std::to_string(f.name().to_int());
			if (vars->find(s) == vars->end()) {
				cout << f.name() << "=> insert " << out_name << endl;
				vars->insert(pair<string, string>(s, out_name));
				*start_var += 1;
			} else {
				out_name = vars->find(s)->second;
			}
		} else {
			out_name = f.name().str();
			vars->insert(pair<string, string>(f.name().str(), out_name));
		}
		out << "-" << out_name << " ";
	} else {
		func_decl f = e.decl();
		string out_name;
		if (f.name().kind() == Z3_INT_SYMBOL) {
			out_name = to_string(*start_var);
			auto s = std::to_string(f.name().to_int());
			if (vars->find(s) == vars->end()) {
				cout << f.name() << "=> insert " << out_name << endl;
				vars->insert(pair<string, string>(s, out_name));
				*start_var += 1;
			} else {
				out_name = vars->find(s)->second;
			}
		} else {
			out_name = f.name().str();
			vars->insert(pair<string, string>(f.name().str(), out_name));
		}
		out << out_name << " ";
	}
}

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

	PBConstraint pbct(col, GEQ, 4);

	pbct.print(false);

	VectorClauseDatabase formula(config);
	// XXX HARDCODED
	AuxVarManager auxvars(17);

	pb2cnf.encode(pbct, formula, auxvars);
	int first_fresh_var = auxvars.getBiggestReturnedAuxVar() + 1;

       /* ofstream out_file;*/
	//out_file.open(argv[1]);
	//out_file << "p cnf " << auxvars.getBiggestReturnedAuxVar() << " " << formula.getClauses().size() << endl;

	expr_vector dnf_expr(z3ctx);

	// First constraint -- CNF
	expr_vector z3and(z3ctx);
	for (auto clause : formula.getClauses()) {
		expr_vector z3or(z3ctx);
		for (auto lit : clause) {
			expr x = z3ctx.bool_const(to_string(abs(lit)).c_str());
			cout << "[1] lit: " << x << endl;
			if (lit > 0)
				z3or.push_back(x);
			else
				z3or.push_back(!x);
		}

		z3and.push_back(mk_or(z3or));
	}
	dnf_expr.push_back(mk_and(z3and));

	col.clear();
	for (int i = 1; i < 6; i++) {
		col.push_back(WeightedLit(i, 3*i / 2 + 1));
	}

	PBConstraint pbct2(col, LEQ, 5);

	pbct2.print(false);

	VectorClauseDatabase formula2(config);
	AuxVarManager auxvars2(first_fresh_var);

	pb2cnf.encode(pbct2, formula2, auxvars2);
	first_fresh_var = auxvars2.getBiggestReturnedAuxVar() + 1;

	// Second constraint -- CNF
	expr_vector z3and2(z3ctx);
	for (auto clause : formula2.getClauses()) {
		expr_vector z3or(z3ctx);
		for (auto lit : clause) {
			expr x = z3ctx.bool_const(to_string(abs(lit)).c_str());
			cout << "lit: " << x << endl;
			if (lit > 0)
				z3or.push_back(x);
			else
				z3or.push_back(!x);
		}

		z3and2.push_back(mk_or(z3or));
	}

	dnf_expr.push_back(mk_and(z3and2));

	//cout << "Negated ineq:" << endl;
	//cout << z3and << endl;

	goal g(z3ctx);
	tactic t1(z3ctx, "simplify");
	tactic t2(z3ctx, "tseitin-cnf");
	tactic t = t1 & t2;
	expr f = mk_or(dnf_expr);
	g.add(f);

	cout << "Goal " << g << endl;

	apply_result r = t(g);
	for (unsigned i = 0; i < r.size(); i++) {
		cout << "subgoal " << i << "\n" << r[i] << "\n";
	}

	int num_vars = 0;
	int num_clauses = 0;
	map<string, string> vars;

	stringstream ss;
	int start_var = first_fresh_var;
	print_dimacs(r[0].as_expr(), &num_clauses, &vars, ss, &start_var);

	for (std::map<string, string>::iterator it=vars.begin(); it!=vars.end(); ++it)
		std::cout << it->first << " => " << it->second << '\n';

	ofstream out_file;
	out_file.open(argv[1]);
	out_file << "p cnf " << vars.size() << " " << num_clauses << endl;
	out_file << ss.rdbuf();

	return 0;
}
