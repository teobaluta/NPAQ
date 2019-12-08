#include "adv_bnn.h"
#include "bnn_prop.h"
#include <stdlib.h>
#include <bits/stdc++.h>

#include "err.h"


int debug = 0;
bool perturb_eq = false;
string encoder = "best";
string OUT_D_IJ = "out_dij";

int parse_opt_args(int start_arg, int argc, char *argv[]) {
	if (strncmp(argv[start_arg], "--", 2) != 0)
		return -EUNKNOWN_CMD_OPTION;

	cout << "argv[" << argv[start_arg] << endl;
	if (argc >= start_arg + 1) {
		if (strncmp(argv[start_arg], "--debug", 5) == 0)
			debug = 1;
		else if (strncmp(argv[start_arg], "--card", 4) == 0)
			encoder = "card";
		else if (strncmp(argv[start_arg], "--bdd", 3) == 0)
			encoder = "bdd";
		else if (strncmp(argv[start_arg], "--equal", 5) == 0)
			perturb_eq = true;
		else {
			cout << "Unrecognized command option " << argv[start_arg] << "." << endl;
			return -EUNKNOWN_CMD_OPTION;
		}
	}

	start_arg++;
	if (argc >= start_arg + 1) {
		if (strncmp(argv[start_arg], "--debug", 7) == 0)
			debug = 1;
		else if (strncmp(argv[start_arg], "--card", 6) == 0)
			encoder = "card";
		else if (strncmp(argv[start_arg], "--bdd", 5) == 0)
			encoder = "bdd";
		else {
			cout << "Unrecognized command option " << argv[start_arg] << "." << endl;
			return -EUNKNOWN_CMD_OPTION;
		}
	}

	return 0;
}

int main(int argc, char *argv[]) {
	// ./bnn2cnf model_dir output_dir debug first_fresh_var

	if (argc < 3 || argc > 8) {
		//cout << "./bnn2cnf model_dir output_dir debug first_fresh_var adv|sparse perturb_file" << endl;
		cout << "./bnn2cnf model_dir output_filename robust|dp|label|diff|fair|canary (--card|--bdd) (--debug)" << endl;
		cout << " - Optional param to select encoding. Default is best." << endl;
		cout << " - If \"robust\" is selected, a perturbation and a path to a concrete input "
			<< "are expected as the next arguments." << endl
			<< "If you wish to specify exactly k perturbations use --equal option" << endl;
		cout << " - If \"label\" is selected, a label_idx is expected as the next argument. "
			<< "This encodes the property the output label idx = label." << endl
			<< "Optionally, you may specify a path to a file with a specific format "
			<< "with constraints over the input vars." << endl;
		cout << " - If \"diff\" is selected, the 2nd model's directory is expected as the next argument." << endl;
		cout << " - If \"fair\" is selected, expecting fairness constraints file and dataset constraints file." << endl;
		cout << " - If \"canary\" is selected, expecting canary value - first 64 bits are 1, rest random." << endl;
		return EXIT_FAILURE;
	}

	string model_dir(argv[1]);
	string out_filename(argv[2]);

	if (argc > 3) {

		int opt = parse_opt_args(3, argc, argv);

		if (opt == 0) {
			// default debug level is 1
			try {
				BNNModel model(model_dir, out_filename);
				int first_fresh_var = model.encode();
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}
			return 0;
		}

		if (strncmp(argv[3], "diff", 4) == 0) {
			if (argc < 5) {
				cout << "./bnn2cnf model1_dir output_filename diff model2_dir <ip constraints>\nExiting..." << endl;
				return EXIT_FAILURE;
			}

			string model2_dir(argv[4]);
			bool add_ip_constraints = false;
			vector<tuple<int, bool>> ip_constraints;
			if (argc > 5) {
				if (strncmp(argv[5], "--", 2) != 0) {
					cout << "specified path to constraints over inputs " << argv[5] << endl;
					add_ip_constraints = true;
				} else {
					int opt = parse_opt_args(5, argc, argv);
					if (opt) {
						cout << "Unrecognized command." << endl;
						return EXIT_FAILURE;
					}
				}
			}

			if (add_ip_constraints) {
				ifstream ip_constraints_f;
				cout << "Reading constraints file " << string(argv[5]) << endl;

				ip_constraints_f.open(string(argv[5]));
				if (ip_constraints_f.is_open()) {
					int ip_var;
					bool ip_var_value;

					// expect format:
					// <location_1> <value_1>
					// <location_2> <value_2>
					// ...
					while (ip_constraints_f >> ip_var >> ip_var_value) {
						if (ip_var_value == 0)
							ip_constraints.push_back(tuple<int, bool>(ip_var, false));
						else
							ip_constraints.push_back(tuple<int, bool>(ip_var, true));
					}
					
					for (const auto &i: ip_constraints)
						cout << get<0>(i) << " " << get<1>(i) << endl;
				} else {
					cout << "Error opening file " << string(argv[5]) << endl;
					return EXIT_FAILURE;
				}

				if (argc > 6) {
					int opt = parse_opt_args(6, argc, argv);
					if (opt) {
						cout << "Unrecognized command." << endl;
						return EXIT_FAILURE;
					}
				}


			}

			// add sparsification
			try {
				//DifferentialNN diff(model_dir, model2_dir, out_filename);
				BNNPropertyEncoder prop_encoder(model_dir, model2_dir, out_filename);

				// THIS ADDS NOT(IP CONSTRAINTS) -- for trojan
				// trying to find very rare events
				if (add_ip_constraints) {
					prop_encoder.encode_dissimilarity(ip_constraints);
				} else {
					prop_encoder.encode_dissimilarity();
				}
				//ret_code = diff.encode();
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}

		} else if (strncmp(argv[3], "dp", 2) == 0) {
			if (argc < 4) {
				cout << "./bnn2cnf model_dir output_filename dp\nExiting..." << endl;
				return EXIT_FAILURE;
			}

			if (argc > 4) {
				int opt = parse_opt_args(5, argc, argv);
				if (opt) {
					cout << "Unrecognized command." << endl;
					return EXIT_FAILURE;
				}
			}

			try {
				BNNPropertyEncoder prop_encoder(model_dir, model_dir, out_filename);

				prop_encoder.encode_dp();
			} catch (int e) {
				cout << "Error " << e << endl;
				return EXIT_FAILURE;
			}

		} else if (strncmp(argv[3], "robust", 6) == 0) {
			if (argc < 6) {
				cout << "./bnn2cnf model_dir output_filename robust perturb_size concrete_input\nExiting..." << endl;
				return EXIT_FAILURE;
			}
			if (argc > 6) {
				int opt = parse_opt_args(6, argc, argv);
				if (opt) {
					cout << "Unrecognized command." << endl;
					return EXIT_FAILURE;
				}
			}

			try {
				BNNPropertyEncoder prop_encoder(model_dir, model_dir, out_filename);
				string concrete_ip_fname(argv[5]);
				vector<bool> concrete_ip;

				ifstream concrete_ip_file;

				concrete_ip_file.open(concrete_ip_fname);
				if (concrete_ip_file.is_open()) {
					int bit;
					while (concrete_ip_file >> bit) {
						if (bit == 0)
							concrete_ip.push_back(false);
						else
							concrete_ip.push_back(true);
					}
				} else {
					cout << "Error opening file " << concrete_ip_fname << endl;
					return EXIT_FAILURE;
				}

				for (int i = 0; i < concrete_ip.size(); i++)
					cout << concrete_ip[i] << " ";
				cout << endl;

				int perturb = atoi(argv[4]);

				prop_encoder.encode_robustness(concrete_ip, perturb, perturb_eq);
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}
		} else if (strncmp(argv[3], "label", 5) == 0) {
			bool add_ip_constraints = false;
			vector<tuple<int, bool>> ip_constraints;

			if (argc < 5) {
				cout << "./bnn2cnf model_dir output_filename label_idx\nExiting..." << endl;
				return EXIT_FAILURE;
			}
			if (argc > 5) {
				if (strncmp(argv[5], "--", 2) != 0) {
					cout << "specified path to constraints over inputs " << argv[5] << endl;
					add_ip_constraints = true;
				}

				ifstream ip_constraints_f;

				ip_constraints_f.open(string(argv[5]));
				if (ip_constraints_f.is_open()) {
					int ip_var;
					bool ip_var_value;

					// expect format:
					// <location_1> <value_1>
					// <location_2> <value_2>
					// ...
					while (ip_constraints_f >> ip_var >> ip_var_value) {
						if (ip_var_value == 0)
							ip_constraints.push_back(tuple<int, bool>(ip_var, false));
						else
							ip_constraints.push_back(tuple<int, bool>(ip_var, true));
					}
					
					for (const auto &i: ip_constraints)
						cout << get<0>(i) << " " << get<1>(i) << endl;
				} else {
					cout << "Error opening file " << string(argv[5]) << endl;
					return EXIT_FAILURE;
				}
			}
			if (argc > 6) {
				int opt = parse_opt_args(6, argc, argv);
				if (opt) {
					cout << "Unrecognized command." << endl;
					return EXIT_FAILURE;
				}
			}

			try {
				BNNPropertyEncoder prop_encoder(model_dir, model_dir, out_filename);
				int label_idx = atoi(argv[4]);

				// first bnn = 0
				if (add_ip_constraints)
					prop_encoder.encode_label(label_idx, ip_constraints);
				else
					prop_encoder.encode_label(label_idx);
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}
		} else if (strncmp(argv[3], "fair", 4) == 0) {
			vector<int> locations;
			vector<bool> values1;
			vector<bool> values2;
			vector<vector<int>> dataset_ct;

			if (argc < 6) {
				cout << "./bnn2cnf model_dir output_filename fair ip_constraints dataset_constraints\nExiting..." << endl;
				return EXIT_FAILURE;
			}

			ifstream ip_constraints_f;

			ip_constraints_f.open(string(argv[4]));
			if (ip_constraints_f.is_open()) {
				// expect format:
				// <locations>
				// <values_1>
				// <values_2>
				string line;
				getline(ip_constraints_f, line);
				cout << line << endl;
				stringstream ss(line);
				string token;

				while (getline(ss, token, ' ')) {
					locations.push_back(atoi(token.c_str()));
				}

				getline(ip_constraints_f, line);
				cout << line << endl;
				stringstream s1(line);

				while (getline(s1, token, ' ')) {
					int v = atoi(token.c_str());
					if (v == 0)
						values1.push_back(false);
					else if (v == 1)
						values1.push_back(true);
					else {
						cout << "WTF. THIS IS 0 or 1 BOOLEAN! ragequit";
						return EXIT_FAILURE;
					}
				}

				getline(ip_constraints_f, line);
				cout << line << endl;
				stringstream s2(line);

				while (getline(s2, token, ' ')) {
					int v = atoi(token.c_str());
					if (v == 0)
						values2.push_back(false);
					else if (v == 1)
						values2.push_back(true);
					else {
						cout << "WTF. THIS IS 0 or 1 BOOLEAN! ragequit";
						return EXIT_FAILURE;
					}
				}
			} else {
				cout << "Error opening file " << string(argv[5]) << endl;
				return EXIT_FAILURE;
			}

			ifstream dataset_ct_f;
			dataset_ct_f.open(string(argv[5]));

			if (dataset_ct_f.is_open()) {

				// expected format
				// sum(vars) < value
				// vars value
				string line;
				while (getline(dataset_ct_f, line)) {
					cout << line << endl;
					vector<int> ct;
					stringstream ss(line);
					string token;

					while (getline(ss, token, ' ')) {
						ct.push_back(atoi(token.c_str()));
					}
					dataset_ct.push_back(ct);
				}

			} else {
				cout << "Error opening file " << string(argv[5]) << endl;
				return EXIT_FAILURE;
			}

			if (argc > 6) {
				int opt = parse_opt_args(6, argc, argv);
				if (opt) {
					cout << "Unrecognized command." << endl;
					return EXIT_FAILURE;
				}
			}

			try {
				BNNPropertyEncoder prop_encoder(model_dir, model_dir, out_filename);
				prop_encoder.encode_fairness2(locations, values1, values2, dataset_ct);
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}
		} else if (strncmp(argv[3], "canary", 6) == 0) {
			cout << "Encoding the log-perplexity value." << endl;
			cout << "Using a hardcoded value for canary" << endl;
			// TODO: read this from file
			vector<int> canary {
				1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
				1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
				1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
				1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,
				1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1,
				1, -1,  1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1
			};

			vector<bool> canary_bool;
			for (auto &i: canary) {
				if (i == 1)
					canary_bool.push_back(true);
				else if (i == -1)
					canary_bool.push_back(false);
				else {
					cout << "canary value " << i << " not recognized." << endl;
				}
			}
			if (argc > 4) {
				int opt = parse_opt_args(4, argc, argv);
				if (opt) {
					cout << "Unrecognized command." << endl;
					return EXIT_FAILURE;
				}
			}

			try {
				BNNPropertyEncoder prop_encoder(model_dir, model_dir, out_filename, true);
				// last 36 bits are the randomness
				prop_encoder.encode_canary(canary_bool, 64);
			} catch (const runtime_error& e) {
				cout << "Exiting gracefully..." << endl << e.what() << endl;
				return EXIT_FAILURE;
			}
		} else {
			cout << "Unrecognized command option " << argv[3] << "." << endl;
			return EXIT_FAILURE;
		}
	} else {
		try {
			BNNModel model(model_dir, out_filename);
			int first_fresh_var = model.encode();
		} catch (const runtime_error& e) {
			cout << "Exiting gracefully..." << endl << e.what() << endl;
			return EXIT_FAILURE;
		}

		return 0;
	}

}

