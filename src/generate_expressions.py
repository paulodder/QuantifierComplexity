import os
import sys
import json
import argparse
import dill
import math
import itertools as it
from pathlib import Path
from itertools import chain
from multiprocessing import Pool, TimeoutError
from functools import partial

import dotenv
import numpy as np

# Add src dir to python path
dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
sys.path.insert(0, str(PROJECT_DIR / "src"))
import operators
import utils
import importlib
import languagegenerator as lg

importlib.reload(lg)
from languagegenerator import LanguageGenerator

sys.argv = [""]

MAX_EXPRESSION_LENGTH = 5
MAX_MODEL_SIZE = 9


DEFAULT_MODEL_SIZE = 5
DEFAULT_EXPRESSION_LENGTH = 5
DEFAULT_NOF_SUBSETS = 3
DEFAULT_JSON_SETUP = "Logical.json"
DEFAULT_DEST_DIR = Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE)


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
        help="Number of subsets to use include if 3: uses A, B, A AND B, if 4: also uses A MINUS B",
    )
    parser.add_argument(
        "--dest_dir",
        "-d",
        type=str,
        default=DEFAULT_DEST_DIR,
        help="Dir to write results to",
    )
    parser.add_argument(
        "--store_at_each_length",
        "-p",
        type=int,
        default=1,
        help="If 1, will store the generated expressions at each round of lengths",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.max_expression_length, args.max_model_size)
    l = lg.LanguageGenerator(
        args.max_model_size,
        args.number_of_subsets,
        args.dest_dir,
        json_path=args.json_setup,
    )
    exps = l.generate_all_sentences(args.max_expression_length)
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
