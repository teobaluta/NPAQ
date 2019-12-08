#!/usr/bin/env python

import arg_parser
import stats

parser = arg_parser.create_parser()
args = parser.parse_args()
args.func(args)

def main():
    pass

if __name__ == "__main__":
    main()
