#!/usr/bin/env python

from __future__ import print_function

import json
from jsonschema import validate


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


class BlkDescription(object):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __str__(self):
        return "in_dim:" + str(self.in_dim) + ",out_dim:" + str(self.out_dim)

class BNNArchDescription(object):
    def __init__(self, json_dict):
        self.name = json_dict['name']
        self.blocks = []

        for blk in json_dict['blocks']:
            print(blk)
            blk = BlkDescription(blk['in_dim'], blk['out_dim'])
            self.blocks.append(blk)

    def __str__(self):
        return "name: " + self.name + "\nblocks:" + str(self.blocks)


# Each type of model that can be parsed has a schema and a class that parses
# that schema
def as_arch_description(json_dict):
    if json_dict['type'] == 'bnn-mnist':
        try:
                validate(json_dict, bnn_schema)
        except Exception as e:
            print('JSON arch description invalid. Exception raised by schema \
                  validator.')
            print(e)

        return BNNArchDescription(json_dict)
