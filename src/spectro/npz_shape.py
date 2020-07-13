#!/bin/python
# print npz archive contents: names, shapes, dtypes

from __future__ import print_function
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize .npz archive contents.",
        epilog="Show npz archive contents: names, shapes, dtypes."
    )
    parser.add_argument("filenames", nargs="+", metavar="file",
                        help="Input filenames.",
    )
    parser.add_argument("pickle", action="store_true", help="Allow pickle")

    args = parser.parse_args()

    indent=4
    indent = " "*(indent-1)
    for fn in args.filenames:
        print(fn,":")
        ds = np.load(fn, allow_pickle=args.pickle)
        for k, d in ds.items():
            print(indent, "{:18s} {:15s} {}".format(k, repr(d.shape), d.dtype))
        
