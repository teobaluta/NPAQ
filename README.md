# NPAQ
Neural Property Approximate Quantifier

## Setup & Requirements

- gcc 5.4.0: To compile the pblib, you need gcc 5.4.0 (newer version gcc 7.4.0
  is not compiling, we have not tested with other versions).
  - Python 2.7
  - (currently z3 requirement but will this dependency will be eliminated)
  - (optional) scalmc (unless only encoding is used)


  First, compile the encoder `cd mlp2cnf; make`. 

  We recommend you use a virtualenv 

  `mkvirtualenv npaq
  pip install -r requirements.txt
  ./setup.sh
  `

## Example Usage

It's always a good idea to check the options with --help.

The general structure is to first specify that we are dealing with BNNs, next to specify neural network size and input space and the dataset.
Then NPAQ offers the following options:
- `encode`, `quant-fair`

`python npaq bnn --arch 1blk_100 --dataset mnist --resize 10,10 quant-robust 2`

The encode option assumes there is a trained model in $results_dir/$model_name/train. For example, for a BNN with architecture of 3 internal blocks with 200 neurons and an output block with 100 neuron-input trained over an input of 28x28 (the default MNIST input) the model_name is bnn_784_3blks_200_100.

I added support to specify the BNN model in a JSON config file. Instead of adding classes to the existing nncrusher/models/bnn.py for every architecture, we can use configs in JSON format. Tested training option only. Command is python nncrusher bnn-mnist --config example_cfg/bnn_1blk.json train.
JSON Schema:

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

python 
