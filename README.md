# NPAQ
Neural Property Approximate Quantifier

## Setup & Requirements

- Python 2.7
- (currently z3 requirement but will this dependency will be eliminated)
- (optional) scalmc (unless only encoding mode is used, you need to have scalmc installed)


First, compile the encoder `cd mlp2cnf; make`. 

We recommend you use a virtualenv 

`mkvirtualenv npaq
pip install -r requirements.txt
./setup.sh

### BNN Training

We used the PyTorch implementation of the binarized neural networks available at
<https://github.com/itayhubara/BinaryNet.pytorch>, hence there might be
differences in the API between current PyTorch versions and the older versions
`torch==1.0.1.post2` and `torchvision==0.2.2.post3`. To use GPU you should make
sure you have the right version for cuda and cuDNN (these should ship with the
pytorch installation). If there are any issues with running on GPU, you may use
the `--no-cuda` flag to disable GPU.

For example, you may write the following command to train a BNN on the MNIST
dataset:

`python npaq bnn --dataset mnist train --no-cuda`

See training help menu with `python npaq bnn train --help`.

## Example Usage

It's always a good idea to check the options with --help.

The general structure is to first specify that we are dealing with BNNs, next to specify neural network architecture (`--arch`), input size (`--resize`) and the dataset (`--dataset`).
NPAQ offers the following options:

- `encode`(to just encode the BNN), `quant-fair`, `quant-robust` and `quant-canary` for trojan attack

`python npaq bnn --arch 1blk_100 --dataset mnist --resize 10,10 quant-robust 2`

The encode option assumes there is a trained model in `models/mnist/` in the format of a `.pt` file (PyTorch model). For example, for a BNN with architecture of 3 internal blocks with 200 neurons and an output block with 100 neuron-input trained over an input of 28x28 (the default MNIST input) the model_name is bnn_784_3blks_200_100.

I added support to specify the BNN model in a JSON config file. Instead of adding classes to the existing nncrusher/models/bnn.py for every architecture, we can use configs in JSON format. Tested training option only. Command is python nncrusher bnn-mnist --config example_cfg/bnn_1blk.json train.
JSON Schema:

```
bnn_schema = {
    "type" : "object",
    "properties" : {
        "model_type" : {"type": "string"},
        "name" : {"type": "string"},
        "blocks" : {"type": "array",
                    "items": {
                        "type" : "object",
                        "properties": {
                            "in_dim": {"type":"integer"},
                            "out_dim": {"type": "integer"},
                            "dropout": {"type": "boolean", "default": "false" }
                        },
                        "required": ["in_dim", "out_dim"]
                    },
                  },
    }
}
```
