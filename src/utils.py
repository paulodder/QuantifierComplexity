import math
import sys
import os
import json
import re
import itertools as it
from pathlib import Path
from functools import partial
from collections import namedtuple, defaultdict, Counter

import dill
import bitarray
import seaborn as sns
import numpy as np
import pandas as pd
import dotenv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pathos.multiprocessing import ProcessingPool as Pool


dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
sys.path.insert(0, str(PROJECT_DIR / "src"))  # add src dir to pyton path
import operators


def exp_len(exp):
    """Given expresion in tuple format, returns expression length"""
    if type(exp) == str:
        return 1
    else:
        return sum(exp_len(x) for x in exp)


def partition_on_length(exps):
    """Given iterable with expressions, returns dict that maps each length to
    list with expressions of that length """
    exp2len = dict(zip(exps, list(map(exp_len, exps))))
    len2exp = dict()
    for exp, length in exp2len.items():
        if length in len2exp:
            len2exp[length].append(exp)
        else:
            len2exp[length] = [exp]
    return len2exp


def prettify_model(m):
    """Prettifies model representation"""
    tups = list(zip(*m))
    out = []
    for a in tups:
        if a[0] and a[1]:
            out.append("AandB")
        elif a[0] and not a[1]:
            out.append("A")
        elif not a[0] and a[1]:
            out.append("B")
        elif not a[0] and not a[1]:
            out.append("M")
    return out


# def load_language_generator_for(max_model_size, experiment):
#     results_dir = Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE)
#     exp_dir_name = make_experiment_dir_name(max_model_size, experiment)
#     with open(
#         Path(results_dir)
#         / Path(exp_dir_name)
#         / "language_generator_object.dill",
#         "rb",
#     ) as f:
#         return dill.load(f)


def same_exp(exp1, exp2):
    """Checks if 2 expressions are syntactically the same"""
    perm_invariant_ops = ["union", "=f", "=", "+", "intersection", "and", "or"]
    if exp1 == exp2:
        return True
    if type(exp1) != type(exp1) or len(exp1) != len(exp2):
        return False
    elif type(exp1) == type(exp2) == tuple:
        if len(exp1) == 3:
            if exp1[0] == exp2[0]:
                if exp1[0] in perm_invariant_ops:
                    return (
                        same_exp(exp1[1], exp2[1])
                        and same_exp(exp1[2], exp2[2])
                    ) or (
                        same_exp(exp1[1], exp2[2])
                        and same_exp(exp1[2], exp2[1])
                    )
                else:
                    return same_exp(exp1[1], exp2[1]) and same_exp(
                        exp1[2], exp2[2]
                    )
        else:
            return exp1[0] == exp2[0] and same_exp(exp1[1], exp2[1])
    elif type(exp1) == str:
        return exp1 == exp2
    else:
        return False


def make_experiment_dir_name(max_model_size, experiment_name):
    """Return name of directory for results of given settings (only depends on
        model size) and experiment name"""
    if experiment_name is not None:
        return f"Experiment={experiment_name}-max_model_size={max_model_size}"
    else:
        return f"max_model_size={max_model_size}"


def return_if_unique(lang_generator, expr):
    meaning = lang_generator.compute_meaning(expr)
    output_type = lang_generator.get_output_type(expr)
    if any(
        lang_generator.same_meaning(meaning, m)
        for m in lang_generator.output_type2expression2meaning[
            output_type
        ].values()
    ):
        pass
    else:  # unique meaning
        return (output_type, expr, meaning)


def generate_unique_expressions_of_len(lang_generator, length):
    type2expressions = dict(
        (t, list()) for t in [bool, operators.SetPlaceholder, int, float]
    )
    output_type2expression2meaning = dict(
        (t, dict()) for t in [bool, operators.SetPlaceholder, int, float]
    )
    with Pool() as p:
        to_apply = partial(return_if_unique, lang_generator)
        for result in p.imap_unordered(
            to_apply,
            lang_generator.yield_expressions_of_len(length),
            chunksize=1000,
        ):
            if result is not None:
                type2expressions[result[0]].append(result[1])
                output_type2expression2meaning[result[0]][result[1]] = result[
                    2
                ]
    return type2expressions, output_type2expression2meaning


def return_meaning_matrix(exps, lang_gen):
    """Given a sorted list with expressions and a language generator, returns
    matrix where the ith row corresponds to the meaning of the ith expression"""
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    out = np.zeros((len(exps), nof_models))
    for i, exp in enumerate(exps):
        out[i, :] = lang_gen.output_type2expression2meaning[bool][exp]
    return out


def divide_possibly_zero(array_a, array_b):
    """return 0 for nans"""
    out = array_a / array_b
    out[np.isnan(out)] = 0
    return out


def element_wise_binary_entropy(prob_df):
    if type(prob_df) is pd.DataFrame:
        return pd.DataFrame(
            data=(
                prob_df.values
                * (np.log(np.divide(1, prob_df.values)) / np.log(2))
                + (
                    (1 - prob_df.values)
                    * np.log(np.divide(1, (1 - prob_df.values)))
                    / np.log(2)
                )
            ),
            index=prob_df.index,
            columns=prob_df.columns,
        ).fillna(0)
    else:
        return pd.Series(
            data=(
                prob_df.values
                * (np.log(np.divide(1, prob_df.values)) / np.log(2))
                + (
                    (1 - prob_df.values)
                    * np.log(np.divide(1, (1 - prob_df.values)))
                    / np.log(2)
                )
            ),
            index=prob_df.index,
        ).fillna(0)


def h(q):
    """Binary entropy func"""
    if q in {0, 1}:
        return 0
    return (q * math.log(1 / q, 2)) + ((1 - q) * math.log(1 / (1 - q), 2))


vectorized_h = np.vectorize(h, otypes=[np.float])


def return_char_array(m):
    out = [0, 0, 0, 0]
    for a, b in zip(*m):
        if a and b:
            out[0] += 1
        elif a:
            out[1] += 1
        elif b:
            out[2] += 1
    return np.array(out)


def compute_char_tuple_matrix(lang_gen):
    """Given lang generator, returns 4 x m matrix, m begin the number of models,
    and each vector representing the characterizing tuple"""
    n_of_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    out = np.zeros((n_of_models, 4), dtype=np.uint8)
    for i, m in enumerate(lang_gen.generate_universe()):
        out[i, :] = return_char_array(m)
    return out


def init_char_tuple_distribution(char_matrix, lang_gen):
    """Returns Counter that maps char tuple to their probability, and another dict
    that maps char tuple to a bool idx array that can be used to select the
    relevant modls"""
    char_tuple2count = Counter()
    char_tuple2idxs = defaultdict(list)
    for i, char_array in enumerate(char_matrix):
        tuple_repr = tuple(char_array)
        char_tuple2count[tuple_repr] += 1
        char_tuple2idxs[tuple_repr].append(i)
    n_of_models = char_matrix.shape[0]
    char_tuple2freq = Counter()
    for char_tuple, count in char_tuple2count.items():
        char_tuple2freq[char_tuple] = count / n_of_models
    char_tuple2bool_idxs = dict()
    for char_tuple, idxs in char_tuple2idxs.items():
        char_tuple2bool_idxs[char_tuple] = np.array(
            [i in idxs for i in range(n_of_models)]
        )
    return (char_tuple2freq, char_tuple2bool_idxs)


def load_expressions_for(experiment_name, max_model_size):
    """Given name of experiment and max model size, loads expressions up to
    greatest length possible (i.e. that have been generated so far)
    :param experiment_name: String containing the name of the json file with
    the corresponding experimental settings
    :param max_model_size: Max size of models on which expressions were
    evaluated
    :returns: List with all expressions in tuple format"""
    experiment_dir = (
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / make_experiment_dir_name(max_model_size, experiment_name)
    )
    to_load = max(
        [f for f in os.listdir(experiment_dir) if "expressions" in f],
        key=lambda x: re.findall("[0-9]+", x)[-1],
    )
    with open(Path(experiment_dir) / to_load, "rb") as f:
        out = dill.load(f)
    return out


def load_lang_gen_for(
    experiment_name, max_model_size, max_expression_length=None
):
    """Given name of experiment and max model size, loads language generator for
    highest possible number of expressions from results dir of that
    experiment

    """
    experiment_dir = (
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / make_experiment_dir_name(max_model_size, experiment_name)
    )
    if max_expression_length is None:
        to_load = max(
            [f for f in os.listdir(experiment_dir) if "lang" in f],
            key=lambda x: re.findall("[0-9]+", x)[-1],
        )
    else:
        for f in os.listdir(experiment_dir):
            if "lang" not in f:
                continue
            if re.findall("[0-9]+", f):
                n = re.findall("[0-9]+", f)[-1]
                if n == str(max_expression_length):
                    to_load = f
                    break
        else:
            print("Could not find file with given settings")
        # print(os.listdir(experiment_dir))
        # print(
        #     [[-1] for f in os.listdir(experiment_dir)]
        # )
        # to_load = next(
        #     f
        #     for f in os.listdir(experiment_dir)
        #     if "lang" in f
        #     and re.findall("[0-9]+", f)[-1] == str(max_expression_length)
        # )
    with open(Path(experiment_dir) / to_load, "rb") as f:
        out = dill.load(f)
    return out


def tuple_format(model):
    """Convert a model to it's tuple format"""
    return tuple((int(a), int(b)) for a, b in zip(*model))
