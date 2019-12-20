import argparse
from multiprocessing import Pool, TimeoutError
import os
import dill
import json
import bitarray
import sys
import numpy as np
import math
from functools import partial
from pathlib import Path

# from Generator import generate_simple_primitive_expressions_with_sets
import itertools as it
from itertools import chain
from collections import defaultdict, namedtuple
import sys

# import
import operators

# from paul_operators import OPERATORS
import imp
import utils


imp.reload(utils)

import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
print(RESULTS_DIR_RELATIVE, JSON_SETUP_DIR_RELATIVE)
# sys.argv = [""]

MAX_EXPRESSION_LENGTH = 12
MAX_MODEL_SIZE = 10


DEFAULT_MODEL_SIZE = 5
DEFAULT_EXPRESSION_LENGTH = 5
DEFAULT_NOF_SUBSETS = 3
DEFAULT_JSON_SETUP = "Logical.json"
# DEFAULT_DEST_DIR = (
#     "/home/paul/Dropbox/courses/quantifiers/SimInf_Quantifiers/MY_RESULTS"
# )


def load_expressions_for(max_len, max_size, experiment="Logical"):
    with open(
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / f"{experiment}_length={max_len}_size={max_size}/expressions.dill",
        "rb",
    ) as f:
        return list(map(expression2tuple, dill.load(f)))


def convert_to_tree(exp):
    if type(exp) is not tuple:
        return Node(exp)
    else:
        return Node(exp[0], children=[convert_to_tree(arg) for arg in exp[1:]])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_expression_length",
        "-l",
        type=int,
        default=MAX_EXPRESSION_LENGTH,
        help="Generate expressions up to this length",
    )
    parser.add_argument(
        "--max_model_size",
        "-m",
        type=int,
        default=MAX_MODEL_SIZE,
        help="Models up to this size will be used to evaluate the meaning of statements",
    )
    parser.add_argument(
        "--json_setup",
        "-j",
        type=str,
        default="Logical.json",
        help="Name of json file to be used with relevant settings",
    )
    parser.add_argument(
        "--number_of_subsets",
        "-s",
        type=int,
        default=DEFAULT_NOF_SUBSETS,
        help="number of subsets to use include if 3: uses A, B, A AND B, if 4: also uses A MINUS B",
    )
    parser.add_argument(
        "--dest_dir",
        "-d",
        type=str,
        default=Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE),
        help="Dir to write results to",
    )
    return parser.parse_args()


def expression2tuple(expr_object):
    out = ()
    if len(expr_object.arg_expressions) == 0:
        # print("parsing", expr_object, type(expr_object.name))
        if type(expr_object.name) is np.float64:
            out = str(round(expr_object.name, 2))
        else:
            out = str(expr_object.name)
        return out
    elif len(expr_object.arg_expressions) == 1:
        return (
            expr_object.name,
            expression2tuple(expr_object.arg_expressions[0]),
        )
    else:
        return (
            expr_object.name,
            expression2tuple(expr_object.arg_expressions[0]),
            expression2tuple(expr_object.arg_expressions[1]),
        )


if __name__ == "__main__":
    args = parse_args()
    print(args.max_expression_length, args.max_model_size)
    l = utils.LanguageGenerator(
        args.max_expression_length,
        args.max_model_size,
        args.number_of_subsets,
        args.dest_dir,
        json_path=args.json_setup,
    )
    exps = l.generate_all_sentences()
    experiment_name = os.path.splitext(os.path.basename(args.json_setup))[0]
    # out_dir = (
    #     Path(args.dest_dir)
    #     / f"{experiment_name}_length={args.max_expression_length}_size={args.max_model_size}"
    # )

    # os.mkdir(out_dir) if not os.path.exists(out_dir) else None
    # with open(out_dir / "expressions.dill", "wb") as f:
    #     dill.dump(exps, f)
    # with open(out_dir / "language_generator.dill", "wb") as f:
    #     dill.dump(l, f)

#     wouters_exps = load_expressions_for(
#         args.max_expression_length, args.max_model_size
#     )
#     unique_meanings(exps, wouters_exps, l)

#     get = lambda size: (
#         np.random.randint(size, size=(sum(4 ** i for i in range(1, size)),)),
#         [np.random.randint(2, size=(i, 4 ** i)) for i in range(1, size)],
#     )
# # a, x = get()
# out = np.zeros((5, 20), dtype=np.uint8)
# c = x.cumsum(0)
# nonzero_x_idxs, nonzero_y_idxs = np.where(c == a)
# x_idxs, mask = np.unique(nonzero_y_idxs, return_index=True)
# out[nonzero_x_idxs[mask], x_idxs] = 1
# a, x = get()
# out = np.zeros((5, 20), dtype=np.uint8)
# c = x.cumsum(0)
# nonzero_x_idxs, nonzero_y_idxs = np.where(c == a)
# x_idxs, mask = np.unique(nonzero_y_idxs, return_index=True)
# out[nonzero_x_idxs[mask], x_idxs] = 1
# out[xs[mask], xs_idx] = 1
