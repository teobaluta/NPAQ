# NPAQ: Quantitative Verification for Neural Nets

NPAQ (Neural Property Approximate Quantifier) is a quantitative verification
tool that can quantify robustness, fairness and trojan attack success for
binarized neural networks (BNNs).  This work is by Teodora Baluta, Shiqi Shen,
Shweta Shinde, Kuldeep S. Meel and Prateek Saxena, as published in [CCS
2019](https://www.comp.nus.edu.sg/~teodorab/papers/NPAQ.pdf). NPAQ relies on
approximate model counting and uses the latest version of
[ApproxMCv3](https://github.com/meelgroup/ApproxMC).

## How to Build

To build on Linux you need the following:

- Python 2.7
- (currently z3 requirement but will this dependency will be eliminated when we
  cleanup more of the code)
- ApproxMC

First, compile the encoder:

```
cd mlp2cnf; make
```

Next, install the other requirements in a virtualenv (here, `npaq`):

```
mkvirtualenv npaq

pip install -r requirements.txt

./setup.sh
```

Unless you want NPAQ to only encode your problem to a CNF formula and not do any
quantification (for whatever reason) you will not need `approxmc`. Otherwise,
please follow the setup instructions for ApproxMC
[here](https://github.com/meelgroup/ApproxMC#how-to-build) and make sure
`approxmc` is in your path.

## Usage Example

It is always a good idea to check the options with `python npaq --help` from the
project's root directory but we will go over the options below.

### Selecting the Architecture, Dataset and Input Size

- Architecture: You can specify the number of blocks and neurons per
block either as a JSON file (see [1]) or you can select from the predefined
ones: `{1blk_100, 2blk_100_50}`
See `npaq/models/bnn.py` for the definitions and project page for details on the [BNN models in
the benchmark](https://teobaluta.github.io/NPAQ/#bnn_models).

- Dataset: By default, the dataset is MNIST (`--dataset mnist`), you can select UCI Adult dataset
  using the option `--dataset uci_adult`.

- Input size: The input size is a pair _width,height_ where  (`--resize`) and the
dataset (`--dataset`). 


### Quantifying Properties

- Encoding to CNF formulas: `encode`. This just encodes the BNN to a CNF
  formula. The encode option assumes there is a trained model in `models/mnist/` in the
format of a `.pt` file (PyTorch model). For example, for a BNN with architecture
of 3 internal blocks with 200 neurons and an output block with 100 neuron-input
trained over an input of 28x28 (the default MNIST input) the file name should be
in the following format `bnn_784_3blks_200_100.pt`. 
Example query:

`python npaq bnn --arch 1blk_100 --dataset uci_adult --resize 10,10 encode`

- Quantify Fairness: `quant-fair constraints_fname`. To quantify fairness you
  need to specify the path to the constraints file.  To reproduce the results in
  the paper, you need to select the right dataset `--dataset uci_adult` and the
  corresponding constraints file from the provided [fairness constraints in the
  benchmarks](https://teobaluta.github.io/npaq). You can specify to just encode
  the property to a CNF formula without quantifying by adding the
  `--just-encode` flag.
Example query:

`python npaq bnn --arch 1blk_100 --dataset uci_adult --resize 1,66 quant-fair uci_adult-marital.txt`

- Quantify Robustness: `quant-robust perturb`. To quantify robustness, specify
  the perturbation size as a L1-distance, i.e., the number of bits different in
  the adversarial example. You can specify to just encode the property to a CNF
  formula without quantifying by adding the `--just-encode` flag.
Example query:

`python npaq bnn --arch 1blk_100 --dataset mnist --resize 10,10 quant-robust 2`

The query above is taking an image from mnist at random. You may specify a concrete input as
following:

`python npaq bnn --encoder card --arch 1blk_100 --dataset mnist --resize 10,10 quant-robust 2 --concrete_ip concrete_inputs --num_samples 1`

- Trojan Attack Success: `quant-canary`.


## Models and BNN Training

We provide the trained models used in the paper as `.pt` files in at our
[project page](https://teobaluta.github.io/NPAQ/#benchmarks). Just copy them in the `models/mnist` folder and specify the
architecture with the `--arch` option 

We used the PyTorch implementation of the binarized neural networks available at
[BinaryNet.pytorch](https://github.com/itayhubara/BinaryNet.pytorch), hence
there might be differences in the API between current PyTorch versions and the
older versions `torch==1.0.1.post2` and `torchvision==0.2.2.post3`. To use GPU
you should make sure you have the right version for cuda and cuDNN (these should
ship with the pytorch installation). If there are any issues with running on
GPU, you may use the `--no-cuda` flag to disable GPU.

You can train your own BNNs using NPAQ. For example, you may write the following
command to train a BNN on the MNIST dataset:

`python npaq bnn --dataset mnist train --no-cuda`

See training help menu with `python npaq bnn train --help`.


## How to Cite

If you use NPAQ, please cite our work.

The benchmarks used in our evaluation can be found [here](https://teobaluta.github.io/NPAQ/). More info on the
project page, [NPAQ](https://teobaluta.github.io/NPAQ).
