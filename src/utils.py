from pathlib import Path
import seaborn as sns
import math
from functools import partial
import dill
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# from multiprocess import Pool, TimeoutError, Process

from pathos.multiprocessing import ProcessingPool as Pool
import operators
import json
import operators
from collections import namedtuple, defaultdict, Counter
import itertools as it
import bitarray
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
# JSON_SETUP_DIR_RELATIVE = "/home/paul/Dropbox/courses/quantifiers/SimInf_Quantifiers/ExperimentSetups/"


# SetDefiner = namedtuple("SetDefiner", "name func")
# SetDefiner.__repr__ = lambda self: self.name
# SetDefiner.__str__ = lambda self: self.name


def partition_on_length(exps):
    desired_len = max(map(exp_len, exps))
    return dict(
        (i, [e for e in exps if exp_len(e) == i])
        for i in range(1, desired_len + 1)
    )


def exp_len(exp):
    if type(exp) == str:
        return 1
    else:
        return sum(exp_len(x) for x in exp)


def prettify_model(m):
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


def load_language_generator_for(max_len, max_size, experiment):
    results_dir = Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE)
    exp_dir_name = self.make_experiment_dir_name()
    with open(
        Path(results_dir)
        / Path(exp_dir_name)
        / "language_generator_object.dill",
        "rb",
    ) as f:
        return dill.load(f)


# def load_expressions_for(max_len, max_size, experiment, who):
#     if who == "paul":
#         results_dir = Path(
#             "/home/paul/Dropbox/courses/quantifiers/SimInf_Quantifiers/MY_RESULTS"
#         )
#     elif who == "wouter":
#         results_dir = Path(
#             "/home/paul/Dropbox/courses/quantifiers/SimInf_Quantifiers/WOUTER_RESULTS"
#         )

#     with open(
#         Path(results_dir)
#         / f"{experiment}_length={max_len}_size={max_size}/expressions.dill",
#         "rb",
#     ) as f:
#         if who == "wouter":
#             return list(map(expression2tuple, dill.load(f)))
#         else:
#             return list(dill.load(f))


def convert_to_tree(exp):
    if type(exp) is not tuple:
        return Node(exp)
    else:
        return Node(exp[0], children=[convert_to_tree(arg) for arg in exp[1:]])


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


def unique_meanings(exps0, exps1, l):
    meaning2exp0 = defaultdict(list)
    meaning2exp1 = defaultdict(list)
    for exp in exps0:
        meaning2exp0[tuple(l.compute_meaning(exp))].append(exp)
    for exp in exps1:
        meaning2exp1[tuple(l.compute_meaning(exp))].append(exp)
    print("Share meaning {:<50}{:<30}".format("My set", "Wouters set"))
    for meaning in set(meaning2exp1).union(set(meaning2exp0)):
        expr0, expr1 = meaning2exp0[meaning], meaning2exp1[meaning]
        if expr0 == expr1:
            continue
        print(
            "Share meaning {:<50}{:<30}".format(
                *tuple(
                    map(
                        str,
                        (
                            # meaning,
                            meaning2exp0[meaning],
                            meaning2exp1[meaning],
                        ),
                    )
                )
            )
        )


perm_invariant_ops = ["union", "=f", "=", "+", "intersection", "and", "or"]


def same_exp1(exp1, exp2):
    if type(exp1) != type(exp2):
        return False
    if exp1 == exp2:
        return True
    if type(exp1) == tuple:
        if len(exp1) != len(exp2):
            return False
        if len(exp1) == 2:
            return exp1[0] == exp2[0] and same_exp1(exp1[1], exp2[1])
        else:

            return (
                exp1[0] == exp2[0]
                and (
                    same_exp1(exp1[1], exp2[2]) and same_exp1(exp1[2], exp2[1])
                )
                or (
                    same_exp1(exp1[1], exp2[1]) and same_exp1(exp1[2], exp2[2])
                )
            )


def same_exp(exp1, exp2):
    # print(exp1, exp2)
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


def make_experiment_dir_name(max_model_size, experiment_name=None):
    """Return name of directory for results of given settings (only depends on
        model size) and experiment name"""
    if experiment_name is not None:
        return f"Experiment={experiment_name}-max_model_size={max_model_size}"
    else:
        return f"max_model_size={max_model_size}"


_generate_expressions_of_len = None


def worker_init(func):
    global _generate_expressions_of_len
    _generate_expressions_of_len = func


def worker(x):
    return _generate_expressions_of_len(x)


def ximap_unordered(func, iterable, processes=None):
    with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
        return p.imap_unordered(worker, iterable)


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


# class LanguageGenerator1:
#     def __init__(self, max_expression_length, max_model_size, json_path=""):
#         self.max_expression_length = max_expression_length
#         self.max_model_size = max_model_size
#         self.N_OF_MODELS = sum(4 ** i for i in range(max_model_size + 1)) - 1
#         self.load_json(json_path)
#         self.set_operators()
#         self.set_size2range()

#     def set_size2range(self):
#         out = dict()
#         cur_cumul = 0
#         for size in range(1, self.max_model_size + 1):
#             out[size] = (
#                 sum(4 ** s for s in range(1, size)),
#                 sum(4 ** s for s in range(1, size + 1)),
#             )
#         self.SIZE2RANGE = out

#     def generate_universe(self):
#         """Given max_size of models, returns all models up to that given size"""
#         for i in range(1, self.max_model_size + 1):
#             for A_tuple in it.product([0, 1], repeat=i):
#                 # A_repr = np.array(A_tuple)
#                 A_repr = bitarray.bitarray(A_tuple)
#                 for B_tuple in it.product([0, 1], repeat=i):
#                     yield (A_repr, bitarray.bitarray(B_tuple))

#     def load_json(self, json_path):
#         if json_path:
#             with open(
#                 Path(PROJECT_DIR) / Path(JSON_SETUP_DIR_RELATIVE) / json_path,
#                 "r",
#             ) as f:
#                 self.json_data = json.load(f)
#         else:
#             self.json_data = None

#     def set_operators(self):
#         """Loads all operators in a dict"""
#         if self.json_data is not None:
#             self.operators = dict(
#                 (op, operators.OPERATORS[op])
#                 for op in self.json_data["operators"]
#             )
#         else:
#             self.operators = operators.OPERATORS

#     def compute_meaning_atom(self, atom):
#         # global N_OF_MODELS
#         atom = self.name2atom[atom.content]
#         if atom.is_constant:
#             if type(atom.content) is int:
#                 # print(atom)
#                 arg_meaning = np.array(
#                     [atom.func("model-invariant")] * self.N_OF_MODELS,
#                     dtype=np.uint8,
#                 )
#                 arg_meaning.flags.writeable = False
#                 # print(arg_meaning)
#             else:
#                 arg_meaning = np.array(
#                     [atom.func("model-invariant")] * self.N_OF_MODELS
#                 )
#                 arg_meaning.flags.writeable = False
#         elif atom.content in [True, False]:
#             arg_meaning = bitarray.bitarray()  # compute_meaning(expr)
#             for model in self.generate_universe():
#                 atom.append(atom.func(model))
#         elif type(atom.content) in [int, float]:
#             arg_meaning = np.array()  # compute_meaning(expr)
#             # arg_meaning.flags.writeable = False
#             for model in self.generate_universe():
#                 arg_meaning = np.append(arg_meaning, atom.func(model))
#             # arg_meaning.flags.writeable = False
#         else:  # set case
#             arg_meaning = []
#             cur_size = 1
#             set_repr = np.zeros(
#                 (cur_size, 4 ** cur_size), dtype=np.uint8
#             )  # base case
#             offset = 0
#             for ind, model in enumerate(self.generate_universe()):
#                 if ind == self.SIZE2RANGE[cur_size][1]:
#                     # print("changing size")
#                     cur_size += 1
#                     offset += set_repr.shape[1]
#                     set_repr.flags.writeable = False
#                     arg_meaning.append(set_repr)
#                     set_repr = np.zeros(
#                         (cur_size, 4 ** cur_size), dtype=np.uint8
#                     )
#                 # print(ind - offset, model, atom.func(model), set_repr)
#                 set_repr[:, ind - offset] = atom.func(model)
#                 # else:
#                 #     print(f"now on from {cur_size} to {cur_size + 1}")
#                 #     # set_repr = np.zeros(
#                 #     #     (cur_size, 4 ** cur_size), dtype=np.uint8
#                 #     # )
#                 #     print(ind, offset, model)
#             else:
#                 print("HERE")
#                 set_repr.flags.writeable = False
#                 arg_meaning.append(set_repr)
#         return arg_meaning

#     def get_output_type(self, arg):
#         if type(arg) is tuple:
#             return self.operators[arg[0]].output_type
#         if type(arg) is float:
#             arg = round(arg, 2)
#         # else:
#         #     match_str = arg
#         return self.atom2output_type[str(arg)]
#         # if match_str in self.atom2output_type.keys():

#     def compute_meaning(self, expr):
#         # print(expr)
#         if type(expr) is not tuple:
#             return self.output_type2expression2meaning[
#                 self.atom2output_type[expr]
#             ][expr]
#         main_operator = self.operators[expr[0]]
#         args = expr[1:]
#         meanings = []
#         for arg in args:
#             output_type = self.get_output_type(arg)
#             try:
#                 meanings.append(
#                     self.output_type2expression2meaning[output_type][arg]
#                 )
#             except:  # should only occur when evaluating a 'new 'meaning not created in this algorithm
#                 new_meaning = self.compute_meaning(arg)
#                 meanings.append(new_meaning)
#         return main_operator.func(*meanings)

#     def same_meaning(self, el0, el1):
#         """Determines if 2 elements have same meaning"""
#         # if self.get_output_type(el0) != self.get_output_type(el1):
#         #     return False
#         if type(el0) != type(el1):
#             out = False
#         elif type(el0) is list:
#             out = all(
#                 np.array_equal(set_repr0, set_repr1)
#                 for set_repr0, set_repr1 in zip(el0, el1)
#             )
#         elif type(el0) is np.ndarray:  # must be bitarray
#             out = np.array_equal(el0, el1)
#         else:
#             print(type(el0))
#             pass  # print(el0, el1, type(el0))
#         # print("same?", el0, el1, out)
#         return out

#     def init_atoms(self):
#         """Generate all atoms and set certain featues accordingly"""
#         # self.expression2meaning = dict()
#         self.output_type2expression2meaning = dict(
#             (output_type, dict())
#             for output_type in [bool, int, float, operators.SetPlaceholder]
#         )
#         output_type2atoms = {
#             int: [],
#             float: [],
#             bool: [],
#             operators.SetPlaceholder: [],
#         }
#         for i in range(0, self.max_model_size + 1, 2):
#             output_type2atoms[int].append(
#                 Atom(str(i), lambda model, i=i: 0 + i, True)
#             )
#         set_name2relevant_index = {"A": 0, "B": 1}
#         for set_name in ["A", "B"]:
#             rel_ind = set_name2relevant_index[set_name]
#             output_type2atoms[operators.SetPlaceholder].append(
#                 Atom(
#                     set_name,
#                     lambda model, rel_ind=rel_ind: np.array(
#                         [int(obj[rel_ind]) for obj in zip(*model)]
#                     ),
#                     False,
#                 )
#             )
#         for q in np.arange(0, 1, 0.1):
#             output_type2atoms[float].append(
#                 Atom(str(round(q, 2)), lambda model, q=q: q, True)
#             )
#         for boolean in [True, False]:
#             output_type2atoms[bool].append(
#                 Atom(str(boolean), lambda model, boolean=boolean: boolean, True)
#             )
#         self.output_type2atoms = output_type2atoms
#         self.atom2output_type = dict()
#         for output_type, atoms in output_type2atoms.items():
#             for atom in atoms:
#                 self.atom2output_type[str(atom.content)] = output_type
#         self.name2atom = dict(
#             (at.content, at)
#             for at in it.chain.from_iterable(output_type2atoms.values())
#         )
#         for at in it.chain.from_iterable(output_type2atoms.values()):
#             meaning_atom = self.compute_meaning_atom(at)
#             self.output_type2expression2meaning[self.get_output_type(at)][
#                 at.content
#             ] = meaning_atom
#             # self.output_type2expression2meaning[atom2output_type][
#             #     str(at.content)
#             # ] = meaning_atom

#     def ignore_or_add(self, expr, length):
#         """Generates meaning tuple for expression for all models of given size checks
#         if already exists if so does not do anything with it
#         """
#         print(expr)
#         meaning = self.compute_meaning(expr)
#         # print(meaning)
#         output_type = self.get_output_type(expr)
#         if any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[output_type].values()
#         ):
#             pass
#         else:  # unique meaning
#             self.output_type2expression2meaning[output_type][expr] = meaning
#             self.length2type2expressions[length][output_type][expr] = meaning

#     def yield_expressions_of_len(self, desired_length):
#         """Yields all expressions of desired_length"""
#         for op_name, operator in self.operators.items():
#             output_type = operator.output_type
#             input_types = operator.input_types
#             if len(input_types) == 1:
#                 input_type = operator.input_types[0]
#                 for sub_expr in self.length2type2expressions[
#                     desired_length - 1
#                 ][input_type]:
#                     yield (op_name, sub_expr)
#             elif len(input_types) == 2:
#                 for j in range(1, desired_length - 1):
#                     arg_len_1 = j
#                     arg_len_2 = desired_length - j - 1
#                     arg_type_1 = input_types[0]
#                     arg_type_2 = input_types[1]
#                     for sub_expr_1 in self.length2type2expressions[arg_len_1][
#                         arg_type_1
#                     ]:
#                         for sub_expr_2 in self.length2type2expressions[
#                             arg_len_2
#                         ][arg_type_2]:
#                             yield (op_name, sub_expr_1, sub_expr_2)

#     def return_if_unique(self, exp):
#         """If unique expression, returns (output_type, exp, meaning)"""
#         meaning = self.compute_meaning(exp)
#         output_type = self.get_output_type(expr)
#         if any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[output_type].values()
#         ):
#             pass
#         else:  # unique meaning
#             return (output_type, expr, meaning)
#             # self.output_type2expression2meaning[output_type][expr] = meaning
#             # self.length2type2expressions[length][output_type][expr] = meaning

#     def generate_all_sentences(self):
#         """Generates all unique meaning sentences up go self.max_quantifier_length and
#         self.max_model_size
#         :returns: List with all unique sentences

#         """
#         self.init_atoms()
#         self.length2type2expressions = defaultdict(lambda: defaultdict(list))
#         for output_type, atoms in self.output_type2atoms.items():
#             self.length2type2expressions[1][output_type] = list(
#                 map(lambda at: at.content, atoms)
#             )
#         for i in range(2, self.max_expression_length + 1):
#             (
#                 type2expressions,
#                 cur_len_output_type2expression2meaning,
#             ) = generate_unique_expressions_of_len(self, i)
#             self.length2type2expressions[i] = type2expressions
#             for (
#                 outp_type,
#                 expression2meaning,
#             ) in cur_len_output_type2expression2meaning.items():
#                 self.output_type2expression2meaning[outp_type].update(
#                     expression2meaning
#                 )
#             # p = Pool()
#             # print(i)
#             # to_apply = partial(self.ignore_or_add, i)
#             # list(
#             #     p.imap_unordered(
#             #         to_apply, list(self.yield_expressions_of_len(i))
#             #     )
#             # )
#             # results = p.imap(to_apply, self.yield_expressions_of_len(i))
#             # print(list(results))
#         #     print(f"LENGTH {i}")
#         #     type2expressions = defaultdict(list)
#         #     for op_name, operator in self.operators.items():
#         #         output_type = operator.output_type
#         #         input_types = operator.input_types
#         #         if len(input_types) == 1:
#         #             input_type = operator.input_types[0]
#         #             for sub_expr in self.length2type2expressions[i - 1][
#         #                 input_type
#         #             ]:

#         #                 expr = (op_name, sub_expr)
#         #                 unique_meaning = self.ignore_or_add(
#         #                     expr, self.max_model_size, length=i
#         #                 )
#         #                 if unique_meaning:
#         #                     type2expressions[output_type].append(expr)
#         #         elif len(input_types) == 2:
#         #             for j in range(1, i - 1):
#         #                 arg_len_1 = j
#         #                 arg_len_2 = i - j - 1
#         #                 arg_type_1 = input_types[0]
#         #                 arg_type_2 = input_types[1]
#         #                 for sub_expr_1 in self.length2type2expressions[
#         #                     arg_len_1
#         #                 ][arg_type_1]:
#         #                     for sub_expr_2 in self.length2type2expressions[
#         #                         arg_len_2
#         #                     ][arg_type_2]:
#         #                         expr = (op_name, sub_expr_1, sub_expr_2)
#         #                         unique_meaning = self.ignore_or_add(
#         #                             expr, args.max_model_size
#         #                         )
#         #                         if unique_meaning:
#         #                             type2expressions[output_type].append(expr)
#         #     self.length2type2expressions[i] = type2expressions
#         #     new = sum(map(len, type2expressions.values()))
#         #     print(f"{new} unique expressions of length {i} added")
#         # return list(self.output_type2expression2meaning[bool])
#         # e
#         # for e in self.output_type2expression2meaning.keys()
#         # if type(e) is tuple and self.operators[e[0]].output_type is bool

#     def unique_meaning(self, exp):
#         """Given expression, checks if it has already encountered an expression with this meaning"""
#         if type(exp) is not tuple:
#             return False
#         meaning = self.compute_meaning(exp)
#         return not any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[
#                 self.get_output_type(exp)
#             ].values()
#         )

#     def exp_with_same_meaning(self, exp):
#         m = self.compute_meaning(exp)
#         # if m in self.expression2meaning.values()
#         for exp, meaning in self.output_type2expression2meaning[
#             self.get_output_type(exp)
#         ].items():
#             if self.same_meaning(meaning, m):
#                 return exp
#         return False


# class LanguageGenerator3:
#     def __init__(
#         self,
#         max_expression_length,
#         max_model_size,
#         number_of_subsets,
#         dest_dir,
#         json_path="",
#     ):
#         self.max_expression_length = max_expression_length
#         self.max_model_size = max_model_size
#         self.number_of_subsets = number_of_subsets
#         self.dest_dir = dest_dir
#         self.N_OF_MODELS = (
#             sum(self.number_of_subsets ** i for i in range(max_model_size + 1))
#             - 1
#         )
#         self.json_path = json_path
#         self.load_json(json_path)
#         self.set_operators()
#         self.set_size2range()

#     def set_size2range(self):
#         out = dict()
#         cur_cumul = 0
#         for size in range(1, self.max_model_size + 1):
#             out[size] = (
#                 sum(self.number_of_subsets ** s for s in range(1, size)),
#                 sum(self.number_of_subsets ** s for s in range(1, size + 1)),
#             )
#         self.SIZE2RANGE = out

#     def write_expressions(self):
#         """Writes current expressions and itself to dill file"""
#         print("INTERIM WRITING")
#         experiment_name = os.path.splitext(os.path.basename(self.json_path))[0]
#         out_dir = (
#             Path(self.dest_dir)
#             / f"{experiment_name}_mp_length={self.max_expression_length}_size={self.max_model_size}"
#         )
#         os.mkdir(out_dir) if not os.path.exists(out_dir) else None
#         current_max_len = max(self.length2type2expressions.keys())
#         current_expressions = list(
#             it.chain.from_iterable(
#                 type2expressions[bool]
#                 for type2expressions in self.length2type2expressions.values()
#             )
#         )
#         with open(
#             out_dir / ("expressions_up_to_length_%s.dill" % current_max_len),
#             "wb",
#         ) as f:
#             dill.dump(current_expressions, f)
#         with open(
#             out_dir
#             / ("language_generator_up_to_length_%s.dill" % current_max_len),
#             "wb",
#         ) as f:
#             dill.dump(self, f)
#         print("DONE INTERIM WRITING")

#     def generate_universe(self):
#         """Given max_size of models, returns all models up to that given size"""
#         for i in range(1, self.max_model_size + 1):
#             for A_tuple in it.product([0, 1], repeat=i):
#                 # A_repr = np.array(A_tuple)
#                 A_repr = bitarray.bitarray(A_tuple)
#                 for B_tuple in it.product([0, 1], repeat=i):
#                     B_repr = bitarray.bitarray(B_tuple)
#                     if self.number_of_subsets == 4:
#                         yield (A_repr, B_repr)
#                     elif self.number_of_subsets == 3:
#                         if all(any(a_b) for a_b in zip(A_repr, B_repr)):
#                             yield (A_repr, B_repr)

#     def load_json(self, json_path):
#         if json_path:
#             with open(
#                 Path(PROJECT_DIR) / Path(JSON_SETUP_DIR_RELATIVE) / json_path,
#                 "r",
#             ) as f:
#                 self.json_data = json.load(f)
#         else:
#             self.json_data = None

#     def set_operators(self):
#         """Loads all operators in a dict"""
#         all_operators = operators.init_operators(
#             self.max_model_size, self.number_of_subsets
#         )
#         print(all_operators)
#         if self.json_data is not None:
#             self.operators = dict(
#                 (op, all_operators[op]) for op in self.json_data["operators"]
#             )
#         else:
#             self.operators = all_operators

#     def compute_meaning_atom(self, atom):
#         # global N_OF_MODELS
#         atom = self.name2atom[atom.content]
#         if atom.is_constant:
#             if type(atom.content) is int:
#                 # print(atom)
#                 arg_meaning = np.array(
#                     [atom.func("model-invariant")] * self.N_OF_MODELS,
#                     dtype=np.uint8,
#                 )
#                 arg_meaning.flags.writeable = False
#                 # print(arg_meaning)
#             else:
#                 arg_meaning = np.array(
#                     [atom.func("model-invariant")] * self.N_OF_MODELS
#                 )
#                 arg_meaning.flags.writeable = False
#         elif atom.content in [True, False]:
#             arg_meaning = bitarray.bitarray()  # compute_meaning(expr)
#             for model in self.generate_universe():
#                 atom.append(atom.func(model))
#         elif type(atom.content) in [int, float]:
#             arg_meaning = np.array()  # compute_meaning(expr)
#             # arg_meaning.flags.writeable = False
#             for model in self.generate_universe():
#                 arg_meaning = np.append(arg_meaning, atom.func(model))
#             # arg_meaning.flags.writeable = False
#         else:  # set case
#             arg_meaning = []
#             cur_size = 1
#             set_repr = np.zeros(
#                 (cur_size, self.number_of_subsets ** cur_size), dtype=np.uint8
#             )  # base case
#             offset = 0
#             for ind, model in enumerate(self.generate_universe()):
#                 if ind == self.SIZE2RANGE[cur_size][1]:
#                     # print("changing size")
#                     cur_size += 1
#                     offset += set_repr.shape[1]
#                     set_repr.flags.writeable = False
#                     arg_meaning.append(set_repr)
#                     set_repr = np.zeros(
#                         (cur_size, self.number_of_subsets ** cur_size),
#                         dtype=np.uint8,
#                     )
#                 # print(ind - offset, model, atom.func(model), set_repr)
#                 set_repr[:, ind - offset] = atom.func(model)
#                 # else:
#                 #     print(f"now on from {cur_size} to {cur_size + 1}")
#                 #     # set_repr = np.zeros(
#                 #     #     (cur_size, 4 ** cur_size), dtype=np.uint8
#                 #     # )
#                 #     print(ind, offset, model)
#             else:
#                 print("HERE")
#                 set_repr.flags.writeable = False
#                 arg_meaning.append(set_repr)
#         return arg_meaning

#     def get_output_type(self, arg):
#         if type(arg) is tuple:
#             return self.operators[arg[0]].output_type
#         if type(arg) is float:
#             arg = round(arg, 2)
#         # else:
#         #     match_str = arg
#         return self.atom2output_type[str(arg)]
#         # if match_str in self.atom2output_type.keys():

#     def compute_meaning(self, expr):
#         # print(expr)
#         if type(expr) is not tuple:
#             return self.output_type2expression2meaning[
#                 self.atom2output_type[expr]
#             ][expr]
#         main_operator = self.operators[expr[0]]
#         args = expr[1:]
#         meanings = []
#         for arg in args:
#             output_type = self.get_output_type(arg)
#             try:
#                 meanings.append(
#                     self.output_type2expression2meaning[output_type][arg]
#                 )
#             except:  # should only occur when evaluating a 'new 'meaning not created in this algorithm
#                 new_meaning = self.compute_meaning(arg)
#                 meanings.append(new_meaning)
#         return main_operator.func(*meanings)

#     def same_meaning(self, el0, el1):
#         """Determines if 2 elements have same meaning"""
#         # if self.get_output_type(el0) != self.get_output_type(el1):
#         #     return False
#         if type(el0) != type(el1):
#             out = False
#         elif type(el0) is list:
#             out = all(
#                 np.array_equal(set_repr0, set_repr1)
#                 for set_repr0, set_repr1 in zip(el0, el1)
#             )
#         elif type(el0) is np.ndarray:  # must be bitarray
#             out = np.array_equal(el0, el1)
#         else:
#             print(type(el0))
#             pass  # print(el0, el1, type(el0))
#         # print("same?", el0, el1, out)
#         return out

#     def init_atoms(self):
#         """Generate all atoms and set certain featues accordingly"""
#         # self.expression2meaning = dict()
#         self.output_type2expression2meaning = dict(
#             (output_type, dict())
#             for output_type in [bool, int, float, operators.SetPlaceholder]
#         )
#         output_type2atoms = {
#             int: [],
#             float: [],
#             bool: [],
#             operators.SetPlaceholder: [],
#         }
#         for i in range(0, self.max_model_size + 1, 2):
#             output_type2atoms[int].append(
#                 Atom(str(i), lambda model, i=i: 0 + i, True)
#             )
#         set_name2relevant_index = {"A": 0, "B": 1}
#         for set_name in ["A", "B"]:
#             rel_ind = set_name2relevant_index[set_name]
#             output_type2atoms[operators.SetPlaceholder].append(
#                 Atom(
#                     set_name,
#                     lambda model, rel_ind=rel_ind: np.array(
#                         [int(obj[rel_ind]) for obj in zip(*model)]
#                     ),
#                     False,
#                 )
#             )
#         for q in np.arange(0, 1, 0.1):
#             output_type2atoms[float].append(
#                 Atom(str(round(q, 2)), lambda model, q=q: q, True)
#             )
#         for boolean in [True, False]:
#             output_type2atoms[bool].append(
#                 Atom(str(boolean), lambda model, boolean=boolean: boolean, True)
#             )
#         self.output_type2atoms = output_type2atoms
#         self.atom2output_type = dict()
#         for output_type, atoms in output_type2atoms.items():
#             for atom in atoms:
#                 self.atom2output_type[str(atom.content)] = output_type
#         self.name2atom = dict(
#             (at.content, at)
#             for at in it.chain.from_iterable(output_type2atoms.values())
#         )
#         for at in it.chain.from_iterable(output_type2atoms.values()):
#             meaning_atom = self.compute_meaning_atom(at)
#             self.output_type2expression2meaning[self.get_output_type(at)][
#                 at.content
#             ] = meaning_atom
#             # self.output_type2expression2meaning[atom2output_type][
#             #     str(at.content)
#             # ] = meaning_atom

#     def ignore_or_add(self, current_length, expr):
#         """Generates meaning tuple for expression for all models of given size checks
#         if already exists if so does not do anything with it
#         """
#         # print(expr)
#         meaning = self.compute_meaning(expr)
#         # print(meaning)
#         output_type = self.get_output_type(expr)
#         if any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[output_type].values()
#         ):
#             return False
#         else:  # unique meaning
#             print("ADDING", expr, meaning)
#             self.output_type2expression2meaning[output_type][expr] = meaning
#             self.length2type2expressions[current_length][output_type].append(
#                 expr
#             )
#             self.output_type2expression2meaning = (
#                 self.output_type2expression2meaning
#             )
#             self.length2type2expressions = self.length2type2expressions
#             return True

#     def yield_expressions_of_len(self, desired_length):
#         """Yields all expressions of desired_length"""
#         for op_name, operator in self.operators.items():
#             output_type = operator.output_type
#             input_types = operator.input_types
#             if len(input_types) == 1:
#                 input_type = operator.input_types[0]
#                 for sub_expr in self.length2type2expressions[
#                     desired_length - 1
#                 ][input_type]:
#                     yield (op_name, sub_expr)
#             elif len(input_types) == 2:
#                 for j in range(1, desired_length - 1):
#                     arg_len_1 = j
#                     arg_len_2 = desired_length - j - 1
#                     arg_type_1 = input_types[0]
#                     arg_type_2 = input_types[1]
#                     for sub_expr_1 in self.length2type2expressions[arg_len_1][
#                         arg_type_1
#                     ]:
#                         for sub_expr_2 in self.length2type2expressions[
#                             arg_len_2
#                         ][arg_type_2]:
#                             yield (op_name, sub_expr_1, sub_expr_2)

#     def return_if_unique(self, exp):
#         """If unique expression, returns (output_type, exp, meaning)"""
#         meaning = self.compute_meaning(exp)
#         output_type = self.get_output_type(expr)
#         if any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[output_type].values()
#         ):
#             pass
#         else:  # unique meaning
#             return (output_type, expr, meaning)

#     def generate_all_sentences(self):
#         """Generates all unique meaning sentences up go self.max_quantifier_length and
#         self.max_model_size
#         :returns: List with all unique sentences

#         """
#         self.init_atoms()
#         self.length2type2expressions = defaultdict(lambda: defaultdict(list))
#         for output_type, atoms in self.output_type2atoms.items():
#             self.length2type2expressions[1][output_type] = list(
#                 map(lambda at: at.content, atoms)
#             )
#         for i in range(2, self.max_expression_length + 1):
#             print(i)
#             # to_apply = partial(selfignore_or_add, i)
#             (
#                 type2expressions,
#                 cur_len_output_type2expression2meaning,
#             ) = generate_unique_expressions_of_len(self, i)
#             self.length2type2expressions[i] = type2expressions
#             for (
#                 outp_type,
#                 expression2meaning,
#             ) in cur_len_output_type2expression2meaning.items():
#                 self.output_type2expression2meaning[outp_type].update(
#                     expression2meaning
#                 )
#             print(
#                 "%s unique expressions of length %s"
#                 % (sum(map(len, type2expressions.values())), i)
#             )
#             self.write_expressions()
#             # with Pool() as p:
#             #     for x in p.imap(to_apply, self.yield_expressions_of_len(i)):
#             #         pass
#         return list(self.output_type2expression2meaning[bool])
#         # e
#         # for e in self.output_type2expression2meaning.keys()
#         # if type(e) is tuple and self.operators[e[0]].output_type is bool

#     def unique_meaning(self, exp):
#         """Given expression, checks if it has already encountered an expression with this meaning"""
#         if type(exp) is not tuple:
#             return False
#         meaning = self.compute_meaning(exp)
#         return not any(
#             self.same_meaning(meaning, m)
#             for m in self.output_type2expression2meaning[
#                 self.get_output_type(exp)
#             ].values()
#         )

#     def write_expressions(self):
#         """Writes current expressions and itself to dill file"""
#         print("INTERIM WRITING")
#         experiment_name = os.path.splitext(os.path.basename(self.json_path))[0]
#         experiment_dir = self.make_experiment_dir_name()
#         out_dir = Path(self.dest_dir) / experiment_dir
#         os.mkdir(out_dir) if not os.path.exists(out_dir) else None
#         current_max_len = max(self.length2type2expressions.keys())
#         current_expressions = list(
#             it.chain.from_iterable(
#                 type2expressions[bool]
#                 for type2expressions in self.length2type2expressions.values()
#             )
#         )
#         with open(
#             out_dir / ("expressions_up_to_length_%s.dill" % current_max_len),
#             "wb",
#         ) as f:
#             dill.dump(current_expressions, f)
#         with open(
#             out_dir
#             / ("language_generator_up_to_length_%s.dill" % current_max_len),
#             "wb",
#         ) as f:
#             dill.dump(self, f)
#         print("DONE INTERIM WRITING")

#     def exp_with_same_meaning(self, exp):
#         m = self.compute_meaning(exp)
#         # if m in self.expression2meaning.values()
#         for exp, meaning in self.output_type2expression2meaning[
#             self.get_output_type(exp)
#         ].items():
#             if self.same_meaning(meaning, m):
#                 return exp
#         return False


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


def quant_for_exps(
    exps,
    lang_gen,
    meaning_matrix,
    char_tuple2freq,
    char_tuple2bool_idxs,
    output_format="dict",
):
    """Compute graded quantity score"""
    # meaning_matrix = return_meaning_matrix(exps, lang_gen)
    for i, exp in enumerate(exps):
        outp_type = lang_gen.get_output_type(exp)
        meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
            outp_type
        ][exp]
    H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    H_Q = vectorized_h(
        np.array(
            (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
            dtype=np.float128,
        )
    )
    for char_tuple, prob in char_tuple2freq.items():
        # relevant_bools = meaning_matrix[char_tuple2bool_idxs[char_tuple]]
        # print(char_tuple)
        relevant_sample_space = meaning_matrix[
            :, char_tuple2bool_idxs[char_tuple]
        ]
        size_of_sample_space = relevant_sample_space.shape[1]
        # print(size_of_sample_space)
        freqs = (
            np.apply_along_axis(sum, 1, relevant_sample_space)
            / size_of_sample_space
        )
        H_Q_given_this_char_tuple = vectorized_h(freqs) * prob
        # print(H_Q_given_this_char_tuple)
        H_Q_given_char_tuple += H_Q_given_this_char_tuple
        # (entropy_of_Q - H_Q_given_char_tuple)
    exp2H_Q = dict((e, q) for e, q in zip(exps, H_Q))
    exp2H_Q_given_char_tuple = dict(
        (e, q) for e, q in zip(exps, H_Q_given_char_tuple)
    )
    if output_format == "dict":
        return dict(
            (exp, q)
            for exp, q in zip(
                exps,
                [
                    ((b - a) / b) if b != 0 else 1
                    for a, b in zip(H_Q_given_char_tuple, H_Q)
                ],
            )
        )
    else:
        pass  # do later
        return None


def divide_possibly_zero(array_a, array_b):
    """return 0 for nans"""
    out = array_a / array_b
    out[np.isnan(out)] = 0
    return out


# def previous_monotonicity_for_exps(
#     exps,
#     lang_gen,
#     meaning_matrix,
#     # char_tuple2freq
#     # , char_tuple2bool_idxs, output_format="dict"
# ):
#     """Compute graded quantity score"""

#     # = return_meaning_matrix(exps, lang_gen)

#     # for i, exp in enumerate(exps):
#     #     outp_type = lang_gen.get_output_type(exp)
#     #     meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
#     #         outp_type
#     #     ][exp]
#     H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
#     nof_models = sum(
#         lang_gen.number_of_subsets ** i
#         for i in range(1, lang_gen.max_model_size + 1)
#     )
#     H_Q = vectorized_h(
#         np.array(
#             (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
#             dtype=np.float128,
#         )
#     )
#     submodel_indicator2freq, signature2idxs = Counter(), defaultdict(list)
#     signature2true_submodel = dict()
#     submodel_indicator_meaning_matrix = np.zeros_like(meaning_matrix)
#     for ind, mod in enumerate(lang_gen.generate_universe()):
#         print(ind)
#         signature = submodel_signature(mod)
#         if signature in signature2true_submodel.keys():
#             previous_signature = signature2true_submodel[signature]
#             new_signature_indicator = np.maximum(
#                 meaning_matrix[:, ind], previous_signature
#             )  # apply_along_axis(
#             #     max,
#             #     1,
#             #     np.mat([]),
#             # )
#             print(new_signature_indicator.shape)
#         else:
#             new_signature_indicator = meaning_matrix[:, ind]
#         print(np.unique(new_signature_indicator))
#         submodel_indicator_meaning_matrix[:, ind] = new_signature_indicator
#         signature2true_submodel[signature] = new_signature_indicator
#         submodel_indicator2freq[0] += new_signature_indicator.astype(int) == 0
#         submodel_indicator2freq[1] += new_signature_indicator.astype(int) == 1
#     submodel_indicator2freq = dict(
#         (k, v / nof_models) for k, v in submodel_indicator2freq.items()
#     )
#     H_Q_given_true_submodel = 0
#     for val, prob in submodel_indicator2freq.items():
#         # p = (meaning_matrix
#         print(val, prob)
#         submodel_meaning_matrix_this_val = (
#             submodel_indicator_meaning_matrix == val
#         )
#         relevant_sample_space_sizes = submodel_meaning_matrix_this_val.sum(1)
#         prob_of_1_in_sample_space = (
#             (meaning_matrix == val) & submodel_meaning_matrix_this_val
#         ).sum(1)
#         p = divide_possibly_zero(
#             prob_of_1_in_sample_space, relevant_sample_space_sizes
#         )
#         #  = meaning_matrix == val
#         #     submodel_meaning_matrix_this_val
#         # ]
#         # if val == 0:
#         #     continue
#         # return (
#         #     submodel_indicator_meaning_matrix,
#         #     relevant_sample_space_sizes,
#         #     meaning_matrix,
#         #     submodel_meaning_matrix_this_val,
#         #     prob_of_1_in_sample_space,
#         # )

#         # meaning_matrix_this_val = meaning_matrix == val
#         # p = (
#         #     relevant_sample_space.sum(1)
#         #     # / .sum()
#         # )
#         H_Q_given_true_submodel_this_val = vectorized_h(p)
#         H_Q_given_true_submodel += prob * H_Q_given_true_submodel_this_val
#     return dict(
#         (e, v)
#         for e, v in zip(
#             exps, divide_possibly_zero((H_Q - H_Q_given_true_submodel), H_Q)
#         )
#     )


def new_monotonicity_for_exps(
    exps,
    lang_gen,
    meaning_matrix,
    # char_tuple2freq
    # , char_tuple2bool_idxs, output_format="dict"
):
    """Compute graded quantity score"""

    # = return_meaning_matrix(exps, lang_gen)

    # for i, exp in enumerate(exps):
    #     outp_type = lang_gen.get_output_type(exp)
    #     meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
    #         outp_type
    #     ][exp]
    H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    H_Q = vectorized_h(
        np.array(
            (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
            dtype=np.float128,
        )
    )
    submodel_indicator2freq = Counter(), dict()
    signature2true_submodel = dict()
    # submodel_indicator_meaning_matrix = np.zeros_like(meaning_matrix)
    a_idxs2ab_idxs2true_on_any = defaultdict(
        lambda: defaultdict(lambda: np.zeros(meaning_matrix.shape[0]))
    )
    signature2inds = dict()
    for ind, mod in enumerate(lang_gen.generate_universe()):
        # print(ind)
        signature = new_submodel_signature(mod)
        # print(signature)
        if signature in signature2inds.keys():
            signature2inds[signature].append(ind)
        else:
            signature2inds[signature] = [ind]
            # ind2signature[ind] = signature2inds
        this_meanings = meaning_matrix[:, ind]
        current_meanings_for_model_type = a_idxs2ab_idxs2true_on_any[
            signature[0]
        ][signature[1]]
        a_idxs2ab_idxs2true_on_any[signature[0]][signature[1]] = np.logical_or(
            this_meanings, current_meanings_for_model_type
        )
    signature2has_true_submodel = dict()
    has_true_submodel_meaning_matrix = np.zeros_like(meaning_matrix)
    # return a_idxs2ab_idxs2true_on_any
    for a_idxs, ab_idxs2true_on_any in a_idxs2ab_idxs2true_on_any.items():
        for ab_idxs, true_on_any in ab_idxs2true_on_any.items():
            signature = (a_idxs, ab_idxs)
            if signature in signature2has_true_submodel.keys():
                has_true_submodel_vector = signature2has_true_submodel[
                    signature
                ]
            else:
                has_true_submodel_vector = true_on_any
                for (
                    other_ab_idxs,
                    other_true_on_any,
                ) in a_idxs2ab_idxs2true_on_any[a_idxs].items():
                    if all(
                        other_ab_idx in ab_idxs
                        for other_ab_idx in other_ab_idxs
                    ):  # submodel_indicator2freq
                        has_true_submodel_vector = np.logical_or(
                            has_true_submodel_vector, other_true_on_any
                        )
                signature2has_true_submodel[
                    signature
                ] = has_true_submodel_vector
            # print(has_true_submodel_vector.shape)
            for relevant_index in signature2inds[signature]:
                has_true_submodel_meaning_matrix[
                    :, relevant_index
                ] = has_true_submodel_vector
                # for a_idxs
    true_submodel2prob = has_true_submodel_meaning_matrix.sum(1) / nof_models
    # H_Q_given_true_submodel = 0
    has_true_submodel_meaning_matrix_sample_space = (
        has_true_submodel_meaning_matrix.astype(int) == 1
    )
    relevant_sample_space_sizes = has_true_submodel_meaning_matrix.astype(
        int
    ).sum(1)
    freq_of_1_in_sample_space = (
        (meaning_matrix == 1) & has_true_submodel_meaning_matrix_sample_space
    ).sum(1)
    prob_of_1_in_sample_space = divide_possibly_zero(
        freq_of_1_in_sample_space, relevant_sample_space_sizes
    )
    # return prob_of_1_in_sample_space
    H_Q_given_true_submodel_this_val = vectorized_h(prob_of_1_in_sample_space)
    H_Q_given_true_submodel = (
        true_submodel2prob * H_Q_given_true_submodel_this_val
    )
    return dict(
        (e, v)
        for e, v in zip(
            exps, divide_possibly_zero((H_Q - H_Q_given_true_submodel), H_Q)
        )
    )

    # for val, prob in submodel_indicator2freq.items():
    #     # p = (meaning_matrix
    #     print(val, prob)
    #     submodel_meaning_matrix_this_val = (
    #         submodel_indicator_meaning_matrix == val
    #     )
    #     relevant_sample_space_sizes = submodel_meaning_matrix_this_val.sum(1)
    #     prob_of_1_in_sample_space = (
    #         (meaning_matrix == val) & submodel_meaning_matrix_this_val
    #     ).sum(1)
    #     p = divide_possibly_zero(
    #         prob_of_1_in_sample_space, relevant_sample_space_sizes
    #     )
    #     #  = meaning_matrix == val
    #     #     submodel_meaning_matrix_this_val
    # ]
    # if val == 0:
    #     continue
    # return (
    #     submodel_indicator_meaning_matrix,
    #     relevant_sample_space_sizes,
    #     meaning_matrix,
    #     submodel_meaning_matrix_this_val,
    #     prob_of_1_in_sample_space,
    # )

    # meaning_matrix_this_val = meaning_matrix == val
    # p = (
    #     relevant_sample_space.sum(1)
    #     # / .sum()
    # )

    # return has_true_submodel_meaning_matrix
    # has_true_submodel = False
    # for existing_signature in signature2true_submodel.keys():
    #     if existing_signature[0] == signature[0] and all(b_idx in signature[1] for b_id
    # # if any(
    #     (
    #         signature[0] == s[0]
    #         and all(ab_idx in s[1] for ab_idx in signature[1])
    #     )
    #     for s in signature2true_submodel.keys()
    # ):
    # previous_signature = max(
    # signature2true_submodel[signature]
    # new_signature_indicator = np.maximum(
    #     meaning_matrix[:, ind], previous_signature
    # )  # apply_along_axis(
    #     max,
    #     1,
    #     np.mat([]),
    # )
    #     print(new_signature_indicator.shape)
    # else:
    #     new_signature_indicator = meaning_matrix[:, ind]
    # print(np.unique(new_signature_indicator))
    # submodel_indicator_meaning_matrix[:, ind] = new_signature_indicator
    # signature2idxs[signature].append(ind)
    # signature2true_submodel[signature] = new_signature_indicator
    # submodel_indicator2freq[0] += new_signature_indicator.astype(int) == 0
    # submodel_indicator2freq[1] += new_signature_indicator.astype(int) == 1
    # submodel_indicator2freq = dict(
    #     (k, v / nof_models) for k, v in submodel_indicator2freq.items()
    # )
    # has_true_submodel_meaning_matrix.sum(1)

    # def new_monotonicity_for_exps(
    #     exps,
    #     lang_gen,
    #     meaning_matrix,
    #     # char_tuple2freq
    #     # , char_tuple2bool_idxs, output_format="dict"
    # ):
    #     """Compute graded quantity score"""

    #     # = return_meaning_matrix(exps, lang_gen)

    #     # for i, exp in enumerate(exps):
    #     #     outp_type = lang_gen.get_output_type(exp)
    #     #     meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
    #     #         outp_type
    #     #     ][exp]
    #     H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
    #     nof_models = sum(
    #         lang_gen.number_of_subsets ** i
    #         for i in range(1, lang_gen.max_model_size + 1)
    #     )
    #     H_Q = vectorized_h(
    #         np.array(
    #             (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
    #             dtype=np.float128,
    #         )
    #     )
    #     submodel_indicator2freq, signature2idxs = Counter(), defaultdict(list)
    #     signature2true_submodel = dict()
    #     submodel_indicator_meaning_matrix = np.zeros_like(meaning_matrix)
    #     for ind, mod in enumerate(lang_gen.generate_universe()):
    #         # print(ind)
    #         signature = submodel_signature(mod)
    #         has_true_submodel = False
    #         for existing_signature in signature2true_submodel.keys():
    #             if existing_signature[0] == signature[0] and all(b_idx in signature[1] for b_id
    #         if any(
    #             (
    #                 signature[0] == s[0]
    #                 and all(ab_idx in s[1] for ab_idx in signature[1])
    #             )
    #             for s in signature2true_submodel.keys()
    #         ):
    #             previous_signature = max(
    #             signature2true_submodel[signature]
    #             new_signature_indicator = np.maximum(
    #                 meaning_matrix[:, ind], previous_signature
    #             )  # apply_along_axis(
    #             #     max,
    #             #     1,
    #             #     np.mat([]),
    #             # )
    #             print(new_signature_indicator.shape)
    #         else:
    #             new_signature_indicator = meaning_matrix[:, ind]
    #         print(np.unique(new_signature_indicator))
    #         submodel_indicator_meaning_matrix[:, ind] = new_signature_indicator
    #         signature2idxs[signature].append(ind)
    #         signature2true_submodel[signature] = new_signature_indicator
    #         submodel_indicator2freq[0] += new_signature_indicator.astype(int) == 0
    #         submodel_indicator2freq[1] += new_signature_indicator.astype(int) == 1
    #     submodel_indicator2freq = dict(
    #         (k, v / nof_models) for k, v in submodel_indicator2freq.items()
    #     )
    #     H_Q_given_true_submodel = 0
    #     for val, prob in submodel_indicator2freq.items():
    #         # p = (meaning_matrix
    #         print(val, prob)
    #         submodel_meaning_matrix_this_val = (
    #             submodel_indicator_meaning_matrix == val
    #         )
    #         relevant_sample_space_sizes = submodel_meaning_matrix_this_val.sum(1)
    #         prob_of_1_in_sample_space = (
    #             (meaning_matrix == val) & submodel_meaning_matrix_this_val
    #         ).sum(1)
    #         p = divide_possibly_zero(
    #             prob_of_1_in_sample_space, relevant_sample_space_sizes
    #         )
    #         #  = meaning_matrix == val
    #         #     submodel_meaning_matrix_this_val
    #         # ]
    #         # if val == 0:
    #         #     continue
    #         # return (
    #         #     submodel_indicator_meaning_matrix,
    #         #     relevant_sample_space_sizes,
    #         #     meaning_matrix,
    #         #     submodel_meaning_matrix_this_val,
    #         #     prob_of_1_in_sample_space,
    #         # )

    #         # meaning_matrix_this_val = meaning_matrix == val
    #         # p = (
    #         #     relevant_sample_space.sum(1)
    #         #     # / .sum()
    #         # )
    #         H_Q_given_true_submodel_this_val = vectorized_h(p)
    #         H_Q_given_true_submodel += prob * H_Q_given_true_submodel_this_val
    #     return dict(
    #         (e, v)
    #         for e, v in zip(
    #             exps, divide_possibly_zero((H_Q - H_Q_given_true_submodel), H_Q)
    #         )
    #     )

    # for
    # return (
    #     submodel_indicator_meaning_matrix,
    #     submodel_indicator2freq,
    #     signature2idxs,
    #     signature2true_submodel,
    # )
    # submodel_indicator2idxs
    # signature2true_submodel
    # submodel_indicator2freq
    #     submodel_indicator2freq[1] += signature2true_submodel[signature]
    # else:
    # new_signature_indicator = np.apply_along_axis(
    #     max, 1, meaning_matrix[:, ind]
    # )

    # submodel_indicator2freq
    # meaning_matrix[:, ind]
    # if signature2true_submodel[signature] == 1:

    # else signature2true_submodel[signature] == 1:

    for char_tuple, prob in char_tuple2freq.items():
        # relevant_bools = meaning_matrix[char_tuple2bool_idxs[char_tuple]]
        # print(char_tuple)
        relevant_sample_space = meaning_matrix[
            :, char_tuple2bool_idxs[char_tuple]
        ]
        size_of_sample_space = relevant_sample_space.shape[1]
        # print(size_of_sample_space)
        freqs = (
            np.apply_along_axis(sum, 1, relevant_sample_space)
            / size_of_sample_space
        )
        H_Q_given_this_char_tuple = vectorized_h(freqs) * prob
        # print(H_Q_given_this_char_tuple)
        H_Q_given_char_tuple += H_Q_given_this_char_tuple
        # (entropy_of_Q - H_Q_given_char_tuple)
    exp2H_Q = dict((e, q) for e, q in zip(exps, H_Q))
    exp2H_Q_given_char_tuple = dict(
        (e, q) for e, q in zip(exps, H_Q_given_char_tuple)
    )
    if output_format == "dict":
        return dict(
            (exp, q)
            for exp, q in zip(
                exps,
                [
                    ((b - a) / b) if b != 0 else 1
                    for a, b in zip(H_Q_given_char_tuple, H_Q)
                ],
            )
        )
    else:
        pass  # do later
        return None


def partition_model_space_modulo(
    lang_gen, signature_function, include_distribution=False
):
    """Map each unique signature to a list that indexes models with corresponding
    signature, note that signature_function must return hashable object """
    sign2idxs = defaultdict(list)
    for ind, mod in enumerate(lang_gen.generate_universe()):
        sign2idxs[signature_function(mod)].append(ind)
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    if include_distribution:
        sign2prob = Counter()
        for sign, idxs in sign2idxs.items():
            sign2prob[sign] = len(idxs) / nof_models
        return sign2idxs, sign2prob
    else:
        return sign2idxs

    # out = dict()
    # nof_models = sum(
    #     lang_gen.number_of_subsets ** i
    #     for i in range(1, lang_gen.max_model_size + 1)
    # )
    # for sign, idxs in sign2idxs.items():
    #     out[sign] = [1 if i in idxs else 0 for i in range(nof_models)]
    # return out


def quant_for_exps(
    exps,
    lang_gen,
    meaning_matrix,
    # char_tuple2freq,
    # char_tuple2bool_idxs,
    output_format="dict",
):
    """Compute graded quantity score"""
    # meaning_matrix = return_meaning_matrix(exps, lang_gen)
    char_tuple2idxs, char_tuple2freq = partition_model_space_modulo(
        lang_gen, quantity_signature, True
    )
    for i, exp in enumerate(exps):
        outp_type = lang_gen.get_output_type(exp)
        meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
            outp_type
        ][exp]
    H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    H_Q = vectorized_h(
        np.array(
            (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
            dtype=np.float128,
        )
    )
    for char_tuple, prob in char_tuple2freq.items():
        # relevant_bools = meaning_matrix[char_tuple2bool_idxs[char_tuple]]
        # print(char_tuple)
        relevant_sample_space = meaning_matrix[:, char_tuple2idxs[char_tuple]]
        size_of_sample_space = relevant_sample_space.shape[1]
        # print(size_of_sample_space)
        freqs = (
            np.apply_along_axis(sum, 1, relevant_sample_space)
            / size_of_sample_space
        )
        H_Q_given_this_char_tuple = vectorized_h(freqs) * prob
        # print(H_Q_given_this_char_tuple)
        H_Q_given_char_tuple += H_Q_given_this_char_tuple
        # (entropy_of_Q - H_Q_given_char_tuple)
    exp2H_Q = dict((e, q) for e, q in zip(exps, H_Q))
    exp2H_Q_given_char_tuple = dict(
        (e, q) for e, q in zip(exps, H_Q_given_char_tuple)
    )
    if output_format == "dict":
        return dict(
            (exp, q)
            for exp, q in zip(
                exps,
                [
                    ((b - a) / b) if b != 0 else 1
                    for a, b in zip(H_Q_given_char_tuple, H_Q)
                ],
            )
        )
    else:
        pass  # do later
        return None


def conservativity_for_exps(
    exps,
    lang_gen,
    meaning_matrix,
    # char_tuple2freq,
    # char_tuple2bool_idxs,
    output_format="dict",
):
    """Compute graded quantity score"""
    # meaning_matrix = return_meaning_matrix(exps, lang_gen)
    char_tuple2idxs, char_tuple2freq = partition_model_space_modulo(
        lang_gen, conservativity_signature, True
    )
    for i, exp in enumerate(exps):
        outp_type = lang_gen.get_output_type(exp)
        meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
            outp_type
        ][exp]
    H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    H_Q = vectorized_h(
        np.array(
            (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
            dtype=np.float128,
        )
    )
    for char_tuple, prob in char_tuple2freq.items():
        # relevant_bools = meaning_matrix[char_tuple2bool_idxs[char_tuple]]
        # print(char_tuple)
        relevant_sample_space = meaning_matrix[:, char_tuple2idxs[char_tuple]]
        size_of_sample_space = relevant_sample_space.shape[1]
        # print(size_of_sample_space)
        freqs = (
            np.apply_along_axis(sum, 1, relevant_sample_space)
            / size_of_sample_space
        )
        H_Q_given_this_char_tuple = vectorized_h(freqs) * prob
        # print(H_Q_given_this_char_tuple)
        H_Q_given_char_tuple += H_Q_given_this_char_tuple
        # (entropy_of_Q - H_Q_given_char_tuple)
    exp2H_Q = dict((e, q) for e, q in zip(exps, H_Q))
    exp2H_Q_given_char_tuple = dict(
        (e, q) for e, q in zip(exps, H_Q_given_char_tuple)
    )
    if output_format == "dict":
        return dict(
            (exp, q)
            for exp, q in zip(
                exps,
                [
                    ((b - a) / b) if b != 0 else 1
                    for a, b in zip(H_Q_given_char_tuple, H_Q)
                ],
            )
        )
    else:
        pass  # do later
        return None


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


def plot_score_distribution_by_length_scatter(
    exp2score, score_name, max_model_size, out_dir
):
    """Plot distribution per length"""
    plt.close("all")
    os.mkdir(out_dir) if not os.path.exists(out_dir) else None
    x, y = [], []
    max_exp_len = 0
    for exp, q in exp2score.items():
        this_exp_len = exp_len(exp)
        x.append(this_exp_len)
        max_exp_len = max(max_exp_len, this_exp_len)
        y.append(q)
    plt.scatter(x, y, alpha=0.05)
    plt.xlabel("length")
    plt.ylabel(f"{score_name.lower()} score")
    plt.title(
        f"{score_name.capitalize()} score distribution by expression length"
    )
    plt.savefig(
        Path(out_dir)
        / f"length-vs-{score_name.lower()}-score_model_size={max_model_size}-up_to_length_{max_exp_len}_check.png"
    )


def plot_score_distribution_by_length_histo(
    exp2score, score_name, max_model_size, out_dir
):
    length2exps = partition_on_length(exp2score.keys())
    max_len = max(length2exps.keys())
    plt.close("all")
    for i, l in enumerate(sorted(length2exps)):
        scores_for_len = [exp2score[e] for e in length2exps[l]]
        if len(set(scores_for_len)) <= 1:
            continue
        else:
            starting_length = l
            fig, ax = plt.subplots(
                max_len - l + 1, 1, sharey=True, sharex=True
            )
            break
    for i, l in enumerate(range(starting_length, max_len + 1)):
        ax[i].set_ylabel("length %s (n=%s)" % (l, len(length2exps[l])))
        try:
            sns.distplot(
                [exp2score[e] for e in length2exps[l]], ax=ax[i],
            )
        except np.linalg.LinAlgError:
            sns.distplot(
                [exp2score[e] for e in length2exps[l]], ax=ax[i], kde=False,
            )
        # ax[i].set_ylabel("length %s (n=%s)" % (l, len(length2exps[l])))

        # if len(set(length2exps[
        # if l <= 2:
        #     continue
        # print(i)
    fig.suptitle(
        f"{score_name.capitalize()} score distribution by length (model_size={max_model_size})",
        fontsize="large",
    )
    # print(max_len - first_valid_length)
    fig.set_size_inches((10, (max_len - 2) * 3),)
    out_fname = (
        out_dir
        / f"{score_name.lower()}-score-distribution-per-length_check.png"
    )
    fig.savefig(out_fname)
    return out_fname

    # for i in sorted(length2exps):
    #     try:
    #         sns.distplot([exp2score[e] for e in length2exps[i]])
    #         first_valid_length = i
    #         print("ey", i)
    #         plt.close("all")
    #         break
    #     except np.linalg.LinAlgError:
    #         continue
    # fig, ax = plt.subplots(
    #     max_len - first_valid_length, 1, sharey=True, sharex=True
    # )
    # print(ax.shape)
    # for i in range(first_valid_length, max_len + 1):
    #     print(i)

    #     sns.distplot(
    #         [exp2score[e] for e in length2exps[i]],
    #         ax=ax[i - first_valid_length],
    #     )


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


def load_lang_gen_for(experiment_name, max_model_size):
    """Given name of experiment and max model size, loads language generator for
    highest possible number of expressions from results dir of that
    experiment

    """
    experiment_dir = (
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / make_experiment_dir_name(max_model_size, experiment_name)
    )
    to_load = max(
        [f for f in os.listdir(experiment_dir) if "lang" in f],
        key=lambda x: re.findall("[0-9]+", x)[-1],
    )
    with open(Path(experiment_dir) / to_load, "rb") as f:
        out = dill.load(f)
    return out


def submodel_signature(m):
    idxs = []
    for i, (a, b) in enumerate(zip(*m)):
        if a == b == True:
            idxs.append(i)
        elif a and not b:
            idxs.append(-i)
    return (sum(m[0]), tuple([x - min(idxs) for x in idxs]))


def new_submodel_signature(m):
    A_idxs = []
    AB_idxs = []
    for i, (a, b) in enumerate(zip(*m)):
        if a == b == True:
            AB_idxs.append(i)
        if a:
            A_idxs.append(i)
    if A_idxs:
        to_subtr = min(A_idxs)
    else:
        to_subtr = 0
        # to_subtr = min((min(A_idxs) or 0, min(AB_idxs) or 0)) or 0
    # except:
    #     print(m)
    return (
        tuple(x - to_subtr for x in A_idxs),
        tuple(x - to_subtr for x in AB_idxs),
    )  # (sum(m[0]), tuple([x - min(idxs) for x in idxs]))


def conservativity_signature(m):
    idxs = []
    for i, (a, b) in enumerate(zip(*m)):
        if a and not b:
            idxs.append(0)
        if a and b:
            idxs.append(1)
    return tuple(idxs)


def quantity_signature(m):
    out = [0, 0, 0, 0]
    for a, b in zip(*m):
        if a and b:
            out[0] += 1
        elif a:
            out[1] += 1
        elif b:
            out[2] += 1
    return tuple(out)

    idxs = []
    for i, (a, b) in enumerate(zip(*m)):
        if a and not b:
            idxs.append(0)
        if a == b == True:
            idxs.append(1)
        # elif a and not b:
        #     idxs.append(-i)
    return tuple([x - min(idxs) for x in idxs])


def scatter(
    exp2score1,
    exp2score2,
    exp2score3,
    axes,
    max_model_size,
    out_dir,
    rotation=30,
):
    sorted_exps = sorted(exp2score1, key=str)
    xs = np.array([exp2score1[e] for e in sorted_exps], dtype=np.float16)
    ys = np.array([exp2score2[e] for e in sorted_exps], dtype=np.float16)
    zs = np.array([exp2score3[e] for e in sorted_exps], dtype=np.float16)
    # [xs, ys] = np.meshgrid(xs, ys)
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # print(xs, ys, zs)
    ax.scatter(xs, ys, zs, alpha=0.05)
    print(len(xs))
    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(axes[2])
    joined_axes = "-vs-".join(axes)
    ax.view_init(30, rotation)
    fig.suptitle(
        f"{joined_axes} score distribution (model_size={max_model_size}) (rotated {rotation} degrees)",
        fontsize="large",
    )
    # print(max_len - first_valid_length)
    fig.set_size_inches((10, 10),)
    zfilled_rotation = str(rotation).zfill(3)
    out_fname = (
        out_dir
        / f"{'-vs-'.join(axes)}-scatter_rotation_{zfilled_rotation}_check.png"
    )
    fig.savefig(out_fname)
    return out_fname
