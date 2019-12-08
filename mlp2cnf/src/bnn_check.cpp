#include "bnn_check.h"


int generate_models(string model_dir, string samples_fname) {
	string out_filename = "temp.dimacs";
	BNNModel bnn(model_dir, out_filename);
	int ret_code = bnn.encode();

	ifstream samples_f;
	cout << "Opening samples file " << samples_fname << endl;
	samples_f.open(samples_fname, std::ifstream::in);

	if (samples_f.is_open()) {
		string sample_raw;
		while (getline(samples_f, sample_raw)) {
			stringstream sample_raw_stream(sample_raw);
			string str;

			getline(sample_raw_stream, str, ':');
			getline(sample_raw_stream, str, ':');
			stringstream str_stream(str);
			cout << str << endl;

			string str_lit;
			vector<CMSat::Lit> assumptions;
			while (getline(str_stream, str_lit, ' ')) {
				int lit = atoi(str_lit.c_str());
				if (lit > 0)
					assumptions.push_back(CMSat::Lit(lit - 1, true));
				else
					assumptions.push_back(CMSat::Lit(-lit + 1, false));
			}

		}
	} else {
		cout << "Error opening " << samples_fname << endl;
	}

	return ret_code;
}
