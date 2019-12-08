#include "gan.h"

string OUTPUT_GAN_0 = "gan_enc_0.dimacs";
string OUTPUT_GAN_1 = "gan_enc_1.dimacs";
string OUTPUT_GAN = "gan_enc.dimacs";

SgmLayer::SgmLayer(string out_dir, vector<string> files) {
	this->files = files;
	this->out_dir = out_dir;

	lin_weight = parseCSV(files[0]);
	lin_bias = parseCSV(files[1]);

	in_size = lin_weight.size();
	// binary classifier 
	if (lin_weight[0].size() == 1)
		out_size = 1;
	else
		// the intermediate variables for the total ordering
		out_size = lin_weight[0].size() * lin_weight[0].size() - lin_weight[0].size();
	//cout << "sgm lin_weight[0].size=" << lin_weight[0].size() << endl;
	//cout << "sgm lin_weight.size()=" << lin_weight.size() << endl;
	//cout << "sgm out_size " << out_size << endl;
}

int SgmLayer::encode(int first_fresh_var, int in_var_start, int out_var_start) {
	PBConfig config = make_shared<PBConfigClass>();
	PB2CNF pb2cnf(config);

	int y;
	if (out_var_start == 0)
		this->out_var_start = in_var_start + lin_weight.size();
	else
		this->out_var_start = out_var_start;

	if (lin_weight[0].size() != 1) {
		cout << "Expecting output size to be 1." << endl;
		exit(1);
	}

	//cout << "sgm bias [" << lin_bias.size() << ", " << lin_bias[0].size() << "]" << endl;
	//cout << "sgm weight [" << lin_weight.size() << ", " << lin_weight[0].size() << "]" << endl;
	vector<WeightedLit> col;

	y = this->out_var_start;
	out_var_end = y;

	int w_minus = 0;
	int sum_aij = 0;
	for (int i = 0; i < lin_weight.size(); i++) {
		//col.push_back(WeightedLit(i + in_var_start, lin_weight[i][0]));
		int var_id = i + in_var_start;

		if (lin_weight[i][0] == 1) {
			WeightedLit x = WeightedLit(var_id, 1);
			col.push_back(x);
		} else if (lin_weight[i][0] == -1) {
			WeightedLit x = WeightedLit(-var_id, 1);
			col.push_back(x);
		}

		if (lin_weight[i][0] == -1)
			// w_minus += abs(round(lin_weight[i][j]));
			w_minus++;
		sum_aij += lin_weight[i][0];
	}
	this->constraints.push_back(col);

	float c = ceil(-lin_bias[0][0]);
	c = ceil(c / 2 + sum_aij / 2) + w_minus;

	PBConstraint pbconstraint(col, GEQ, c);
	pbconstraint.addConditional(y);

	if (debug)
		pbconstraint.print(false);
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
		this->cnf_formula.push_back(clause);

	PBConstraint pbct(col, LEQ, c - 1);
	pbct.addConditional(-y);

	if (debug)
		pbct.print(false);
	VectorClauseDatabase f(config);
	AuxVarManager auxv(first_fresh_var);

	pb2cnf.encode(pbct, f, auxv);
	first_fresh_var = auxv.getBiggestReturnedAuxVar() + 1;
	if (debug)
		cout << "Encoded ^; first_fresh_var = " << first_fresh_var << endl;

	for (auto clause : f.getClauses())
		this->cnf_formula.push_back(clause);


	return first_fresh_var;
}

GeneratorModel::GeneratorModel(string model_dir, string out_dir) {
	DIR *dir;
	struct dirent *ent;
	num_internal_blocks = 0;

	this->out_dir = out_dir;
	this->model_dir = model_dir;

	if ((dir = opendir(model_dir.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string d_name(ent->d_name);

			if (d_name.compare(0, 3, "blk") == 0) {
				num_internal_blocks += 1;
				vector<string> files = filename(this->model_dir, num_internal_blocks);
				//cout << "Blk" << num_internal_blocks << " : " << files << endl;
				BNNBlock blk(files);
				blocks.push_back(blk);
			}
		}
		closedir(dir);
	} else {
		/* could not open directory */
		perror ("");
		// XXX add something like this in the other constructors too
		throw std::runtime_error("Could not open file " + model_dir);
	}
}

int GeneratorModel::encode(int in_var_start, int first_fresh_var) {
	int biggest_aux_var;
	// layer 0 output variables start after the input and output
	// variables
	in_size = blocks[0].in_size;
	//cout << "[generator] input size = " << in_size << endl;
	// layer 0 output variables start after the input and output
	// variables
	out_var_start = blocks[0].in_size + blocks[blocks.size()-1].out_size + 2;
	//cout << "[generator] out_var_start = " << out_var_start << endl;

	write_to_meta(this->out_dir, to_string(first_fresh_var));
	//cout << "first_fresh_var = " << first_fresh_var << "; out var " << out_var_start << endl;

	//cout << "generator num_internal_blocks " << num_internal_blocks << endl;
	for (int i = 0; i < num_internal_blocks; i++) {
		//cout << "Encoding BLK" << i << endl;
		if (i == 0) {
			biggest_aux_var = blocks[i].encode(first_fresh_var, in_var_start, out_var_start);
			write_to_meta(this->out_dir, "blk" + to_string(i) + ": " + \
				      to_string(in_var_start) + " " + to_string(in_var_start + blocks[i].in_size) + " " +\
				      to_string(blocks[i].out_var_start) + " " + \
				      to_string(blocks[i].out_var_end),
				      ios::app);
		} else if (i == blocks.size() - 1) {
			biggest_aux_var = blocks[i].encode(first_fresh_var, in_var_start, in_size + 2);
			write_to_meta(this->out_dir, "blk" + to_string(i) + ": " + to_string(in_var_start) + " " + \
				      to_string(in_var_start + blocks[i].in_size) + \
				      to_string(blocks[i].out_var_start) + " " + \
				      to_string(blocks[i].out_var_end),
				      ios::app);
		} else {
			biggest_aux_var = blocks[i].encode(first_fresh_var, in_var_start);
			write_to_meta(this->out_dir, "blk" + to_string(i) + ": " + to_string(in_var_start) + " " + \
				      to_string(blocks[i].in_size + in_var_start) + " " + \
				      to_string(blocks[i].out_var_start) + " " + \
				      to_string(blocks[i].out_var_end),
				      ios::app);
		}
		in_var_start = blocks[i].out_var_start;
		first_fresh_var = biggest_aux_var;
		//cout << "in_var_start for blk " << i << ": " << in_var_start << endl;
		//cout << "biggest aux var " << biggest_aux_var << endl;
	}

	return biggest_aux_var;
}

DiscriminatorModel::DiscriminatorModel(string model_dir, string out_dir) {
	DIR *dir;
	struct dirent *ent;
	num_internal_blocks = 0;

	this->out_dir = out_dir;
	this->model_dir = model_dir;

	if ((dir = opendir(model_dir.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string d_name(ent->d_name);

			//cout << d_name << endl;
			if (d_name.compare(0, 3, "blk") == 0) {
				num_internal_blocks += 1;
				vector<string> files = filename(this->model_dir, num_internal_blocks);
				//cout << "Blk1 : " << files << endl;
				BNNBlock blk(files);
				this->blocks.push_back(blk);
			}
		}
		closedir(dir);
	} else {
		/* could not open directory */
		perror ("");
		return;
	}
	sgm_layer = new SgmLayer(out_dir, filename(this->model_dir, 0, false));
}

int DiscriminatorModel::encode(int in_var_start, int first_fresh_var, int out_var_start) {
	int biggest_aux_var;
	
	write_to_meta(this->out_dir, to_string(first_fresh_var));

	int initial_in_var_start = in_var_start;
	for (int i = 0; i < num_internal_blocks; i++) {
		//cout << "Encoding BLK" << i << endl;
		if (i == 0) {
			biggest_aux_var = this->blocks[i].encode(first_fresh_var, in_var_start, out_var_start);

			write_to_meta(this->out_dir, "blk" + to_string(i) + ": " + \
				      to_string(in_var_start) + " " + to_string(in_var_start + blocks[i].in_size) + " " + \
				      to_string(blocks[i].out_var_start) + " " + to_string(blocks[i].out_var_end),
				      ios::app);
		} else {
			biggest_aux_var = this->blocks[i].encode(first_fresh_var, in_var_start);
			write_to_meta(this->out_dir, "blk" + to_string(i) + ": " + \
				      to_string(in_var_start) + " " + to_string(in_var_start + blocks[i].in_size) + " " + \
				      to_string(blocks[i].out_var_start) + " " + to_string(blocks[i].out_var_end),
				      ios::app);
		}
		in_var_start = blocks[i].out_var_start;
		first_fresh_var = biggest_aux_var;
	}

	// special encoding for the last layer + sigmoid
	/*biggest_aux_var = sgm_layer->encode(first_fresh_var, in_var_start,*/
					    //blocks[blocks.size() - 1].out_var_end + 1);

	biggest_aux_var = sgm_layer->encode(first_fresh_var, in_var_start, initial_in_var_start - 1);

	//cout << "out var end " << sgm_layer->out_var_end << endl;

	write_to_meta(this->out_dir, "out: " + to_string(in_var_start) + " " + \
		      to_string(in_var_start + sgm_layer->in_size) +  " " +\
		      to_string(sgm_layer->out_var_start) + " " + \
		      to_string(sgm_layer->out_var_end),
		      ios::app);

	return biggest_aux_var;
}


GAN::GAN(string model_dir, string out_dir) {
	this->model_dir = model_dir;
	this->out_dir = out_dir;
	int err;
	char separator;
#ifdef _WIN32
	separator = '\\';
#else
	separator = '/';
#endif

	string out_filename_0 = this->out_dir + "/" + OUTPUT_GAN_0;
	string out_filename_1 = this->out_dir + "/" + OUTPUT_GAN_1;

	//cout << "[Warning] Overwriting saved CNF formulas in " << out_dir;
	if (remove(out_filename_0.c_str()) != 0) {
		//perror( "Error deleting file");
	}

	if (remove(out_filename_1.c_str()) != 0) {
		//perror( "Error deleting file");
	}

	string gen_model = this->model_dir + separator + "generator/model";
	//string gen_out_dir = this->out_dir + separator + "generator/encoding";

	string dis_model = this->model_dir + separator + "discriminator/model";
	//string dis_out_dir = this->out_dir + separator + "discriminator/encoding";

	char *cstr = new char[out_dir.length() + 1];
	strcpy(cstr, out_dir.c_str());

	err = ensure_dir(cstr);
	if (err < 0) {
		// XXX add something like this in the other constructors too
		throw std::runtime_error("Could create dir " + out_dir);
	}
	delete [] cstr;

       /* char *cstr = new char[gen_out_dir.length() + 1];*/
	//strcpy(cstr, gen_out_dir.c_str());

	//err = ensure_dir(cstr);
	//if (err < 0) {
		//// XXX add something like this in the other constructors too
		//throw std::runtime_error("Could create dir " + gen_out_dir);
	//}
	//delete [] cstr;

	//char *dstr = new char[dis_out_dir.length() + 1];
	//strcpy(dstr, dis_out_dir.c_str());

	//err = ensure_dir(dstr);
	//if (err < 0) {
		//// XXX add something like this in the other constructors too
		//throw std::runtime_error("Could create dir " + dis_out_dir);
	//}
	//delete [] dstr;

	/*generator = new GeneratorModel(gen_model, gen_out_dir);*/
	/*discriminator = new DiscriminatorModel(dis_model, dis_out_dir);*/

	generator = new GeneratorModel(gen_model, out_dir);
	discriminator = new DiscriminatorModel(dis_model, out_dir);

}

int GAN::encode() {
	int input_var_start = 0;
	int reserved_vars = 1;
	// write out = 1 and out = 0 for the two CNFs
	ofstream out_file_0;
	ofstream out_file_1;
	string out_filename_0;
	string out_filename_1;
	// write to output without constraint on output variable
	ofstream out_file;
	string out_filename;
	// XXX super ugly; refactor into method in utils to write CNF in dimacs
	// format
	out_filename_0 = this->out_dir + "/" + OUTPUT_GAN_0;
	out_file_0.open(out_filename_0);
	out_filename_1 = this->out_dir + "/" + OUTPUT_GAN_1;
	out_file_1.open(out_filename_1);
	out_filename = this->out_dir + "/" + OUTPUT_GAN;
	out_file.open(out_filename);

	// reserve variables for generator and discriminator
	// GENERATOR RESERVE
	int in_size = generator->blocks[0].in_size;
	if (debug)
		cout << "[generator] input variables [1 - " <<  in_size << "]" << endl;
	// layer 0 output variables start after the input and output variables
	int out_var_start = generator->blocks[0].in_size + 2;

	if (debug)
		cout << "[generator] reserved for input 1 - in_size and and output 1 variable and intermediate variables start from " << out_var_start << endl;

	// init reserved variables; account for input and output vars
	reserved_vars = out_var_start;

	// reserve vars for each of the generator layer's intermediate variables
	for (int i = 0; i < generator->num_internal_blocks; i++)
		reserved_vars += generator->blocks[i].out_size;

	if (debug)
		cout << "[generator] need " << reserved_vars << " vars" << endl;
	// discriminator input vars are overlapping with the generator output
	int d_out_var_start = reserved_vars;

	// discriminator input is overlapping with the generator variables
	// reserve for intermediate variables
	for (int i = 0; i < discriminator->num_internal_blocks; i++)
		reserved_vars += discriminator->blocks[i].out_size;

	if (debug)
		cout << "[discriminator] output variables (this is where the fresh" << "variables start for generator and discriminator) = " << reserved_vars << endl;

	// in_var_start = 1, but fresh_var start with reserved_vars
	int first_fresh_var = generator->encode(1, reserved_vars);

	if (debug) {
		cout << "generator first_fresh_var " << first_fresh_var << endl;
		cout << "Finished encoding generator. Proceed to discriminator" << endl;
	}
	// account for output variable of the discriminator
	int d_in_var_start = generator->in_size + 2;

	if (debug) {
		cout << "[discriminator] in_var_start " << d_in_var_start << endl;
		cout << "[discriminator] d_out_var_start " << d_out_var_start << endl;
		cout << "first_fresh_var " << first_fresh_var << endl;
	}

	int biggest_aux_var = discriminator->encode(d_in_var_start, first_fresh_var, d_out_var_start);

	if (debug) {
		cout << "reserved vars = " << reserved_vars << endl;
	}

	if (out_file_1.is_open() && out_file_0.is_open() && out_file.is_open()) {
		int total_clauses = 0;
		for (int i = 0; i < generator->num_internal_blocks; i++) {
			total_clauses += generator->blocks[i].cnf_formula.size();
		}

		for (int i = 0; i < discriminator->num_internal_blocks; i++) {
			total_clauses += discriminator->blocks[i].cnf_formula.size();
		}

		total_clauses += discriminator->sgm_layer->cnf_formula.size();

		out_file << "p cnf " << biggest_aux_var - 1 << " " << total_clauses << endl;
		// adding a clause for the clause encoding discriminator output
		// = 1 or the discriminator output = 0
		total_clauses += 1;

		out_file_0 << "p cnf " << biggest_aux_var - 1 << " " << total_clauses << endl;
		out_file_1 << "p cnf " << biggest_aux_var - 1 << " " << total_clauses << endl;

		for (int i = 1; i < d_in_var_start + discriminator->blocks[0].in_size; i++) {
			string str = "c ind ";
			int j = 0;
			for (; j < 9; j++) {
				if (i + j < d_in_var_start + discriminator->blocks[0].in_size)
					str += to_string(i + j) + " ";
				else {
					str += " ";
					break;
				}
			}
			i += j - 1;
			out_file_0 << str << "0" << endl;
			out_file_1 << str << "0" << endl;
			out_file << str << "0" << endl;
		}

		for (int i = 0; i < generator->num_internal_blocks; i++)
			for (auto clause : generator->blocks[i].cnf_formula) {
				for (auto lit : clause) {
					out_file_0 << lit << " ";
					out_file_1 << lit << " ";
					out_file << lit << " ";
				}
				out_file_0 << "0" << endl;
				out_file_1 << "0" << endl;
				out_file << "0" << endl;
			}

		for (int i = 0; i < discriminator->num_internal_blocks; i++)
			for (auto clause : discriminator->blocks[i].cnf_formula) {
				for (auto lit : clause) {
					out_file_0 << lit << " ";
					out_file_1 << lit << " ";
					out_file << lit << " ";
				}
				out_file_0 << "0" << endl;
				out_file_1 << "0" << endl;
				out_file << "0" << endl;
			}

		for (auto clause : discriminator->sgm_layer->cnf_formula) {
				for (auto lit : clause) {
					out_file_0 << lit << " ";
					out_file_1 << lit << " ";
					out_file << lit << " ";
				}
				out_file_0 << "0" << endl;
				out_file_1 << "0" << endl;
				out_file << "0" << endl;
			}

		out_file_0 << -(generator->in_size + 1) << " 0" << endl;
		out_file_1 << generator->in_size + 1 << " 0" << endl;

		out_file_0.close();
		out_file_1.close();
		out_file.close();
	} else {
		cout << "Error opening output file " << out_filename_0 << " or " << out_filename_1 << " or " << out_filename << endl;
		throw std::runtime_error("Could not open file " + model_dir);
	}

	return 0;
}
