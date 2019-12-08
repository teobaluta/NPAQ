#include "bnn.h"

class AdvBNN {
      public:
	BNNModel *bnn;
	BNNModel *perturb_bnn;
	string out_dir;
	string model_dir;
	ofstream out_file;
	set<int> perturb;

	AdvBNN(string model_dir, string out_dir, string perturb_filename);

	int encode();
};
