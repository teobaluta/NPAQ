#include "utils.h"

string VARS = "metainfo";

// TODO make this nicer into a separate csv reader
vector<vector<float>> parseCSV(string filename, char delimiter, int debug) {
	ifstream csvfile;
	string line;
	vector<vector<float>> matrix;

	csvfile.open(filename);

	if (csvfile.is_open()) {
		if (debug)
			cout << "Opened " << filename << endl;

		while (getline(csvfile, line)) {
			vector<float> matrix_line;
			string token;
			istringstream tokenstream(line);
			while (getline(tokenstream, token, delimiter)) {
				matrix_line.push_back(stold(token));
			}
			matrix.push_back(matrix_line);
			if (debug) {
				for (auto f : matrix_line) {
					cout << f <<  " ";
				}
				cout << endl;
			}
		}
	} else {
		throw std::runtime_error("Error opening " + filename);
	}

	return matrix;
}

/*
 * POSIX - checking directory exists
 * and creating if it does not exist
 */
int ensure_dir(char *path) {
	DIR *dir = opendir(path);
	if (dir == NULL) {
		if (errno == ENOENT) {
			cout << "Directory " << path << " does not exist." << endl;
			const int dir_err = mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (dir_err == -1) {
				cout << "Error creating " << path << endl;
				exit(1);
			}
		} else {
			return errno;
		}
	}

	return 0;
}

vector<string> filename(string path, int number, bool internal_blk) {
	char separator;
#ifdef _WIN32
	separator = '\\';
#else
	separator = '/';
#endif
	string suffix;
	vector<string> files;
	if (internal_blk == true) {
		suffix = path + separator + "blk"  + to_string(number) + separator;
		files.push_back(suffix + "lin_weight.csv");
		files.push_back(suffix + "lin_bias.csv");
		files.push_back(suffix + "bn_weight.csv");
		files.push_back(suffix + "bn_bias.csv");
		files.push_back(suffix + "bn_mean.csv");
		files.push_back(suffix + "bn_var.csv");
	} else {
		suffix = path + separator + "out_blk" + separator;
		files.push_back(suffix + "lin_weight.csv");
		files.push_back(suffix + "lin_bias.csv");
	}

	return files;
}

int write_to_meta(string out_dir, string what, std::ios_base::openmode mode) {
	ofstream meta_file;
	string meta_filename(out_dir + "." + VARS);
	meta_file.open(meta_filename, mode);
	if (!meta_file.is_open()) {
		cout << "Error opening meta output file " << meta_filename << endl;
		return 1;
	}

	meta_file << what << endl;
	meta_file.close();
}


