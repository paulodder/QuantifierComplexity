import dotenv
import sys
import os
import sys
import json

import dill
import bitarray
import numpy as np
import itertools as it
from pathlib import Path
from collections import namedtuple
from collections import defaultdict
from utils import make_experiment_dir_name

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
sys.path.insert(0, str(PROJECT_DIR / "src"))
import operators


Atom = namedtuple("Atom", "content func is_constant")
Atom.__repr__ = lambda self: str(self.content)

Expression = namedtuple("Expression", "content func")
Expression.__repr__ = lambda self: str(self.content)


class LanguageGenerator:
    """Object that can generate all unique sentences according to the provided
    settings"""

    def __init__(
        self,
        max_model_size,
        dest_dir=Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE),
        store_at_each_length=1,
        json_path="",
    ):
        """Initializes LanguageGenerator object
        :param max_model_size: The size of models on which expressions will be
        evaluated
        :param dest_dir: Directory in which to store results
        :param store_at_each_length: If 1, stores expressions and itself at
        every length
        :param json_path: Name of json file """

        self.max_model_size = max_model_size
        self.dest_dir = dest_dir
        self.store_at_each_length = store_at_each_length
        self.json_path = json_path
        self._load_json(json_path)
        self.N_OF_MODELS = (
            sum(self.number_of_subsets ** i for i in range(max_model_size + 1))
            - 1
        )
        self._set_output_dirs()
        self._set_operators()
        self._set_size2range()
        self.meaning_matrix = None

    def _set_output_dirs(self):
        """Set output directories"""
        self.results_dir = Path(self.dest_dir) / make_experiment_dir_name(
            self.max_model_size, self.experiment_name
        )
        self.figure_dir = self.results_dir / "figures"
        os.mkdir(self.results_dir) if not os.path.exists(
            self.results_dir
        ) else None
        os.mkdir(self.figure_dir) if not os.path.exists(
            self.figure_dir
        ) else None

    def _set_size2range(self):
        """Set size2range feature that maps the size of the model to a tuple
        with 2 elements that denote the range of an array that stores all
        models of that size"""
        out = dict()
        cur_cumul = 0
        for size in range(1, self.max_model_size + 1):
            out[size] = (
                sum(self.number_of_subsets ** s for s in range(1, size)),
                sum(self.number_of_subsets ** s for s in range(1, size + 1)),
            )
        self.SIZE2RANGE = out

    def generate_universe(self):
        """Given max_size of models, returns generator object that yields all
        models up to that given size, sorted lexicographically"""
        for i in range(1, self.max_model_size + 1):
            for A_tuple in it.product([0, 1], repeat=i):
                A_repr = bitarray.bitarray(A_tuple)
                for B_tuple in it.product([0, 1], repeat=i):
                    B_repr = bitarray.bitarray(B_tuple)
                    if self.number_of_subsets == 4:
                        yield (A_repr, B_repr)
                    elif self.number_of_subsets == 3:
                        if all(any(a_b) for a_b in zip(A_repr, B_repr)):
                            yield (A_repr, B_repr)

    def _load_json(self, json_path):
        """Loads JSON file from json directory and sets settings accordingly"""
        if json_path:
            with open(
                Path(PROJECT_DIR) / Path(JSON_SETUP_DIR_RELATIVE) / json_path,
                "r",
            ) as f:
                self.json_data = json.load(f)
                self.experiment_name = self.json_data["name"]
                self.number_of_subsets = self.json_data["number_of_subsets"]
        else:
            self.json_data = None
            self.experiment_name = None

    def _set_operators(self):
        """Loads all operators in a dict"""
        all_operators = operators.init_operators(
            self.max_model_size, self.number_of_subsets
        )
        if self.json_data is not None:
            self.operators = dict(
                (op, all_operators[op]) for op in self.json_data["operators"]
            )
        else:
            self.operators = all_operators

    def _compute_meaning_atom(self, atom):
        """Given an Atom, computes the 'meaning' or 'representation' of that
        atom"""
        atom = self.name2atom[atom.content]
        if atom.is_constant:
            if type(atom.content) is int:
                arg_meaning = np.array(
                    [atom.func("model-invariant")] * self.N_OF_MODELS,
                    dtype=np.uint8,
                )
                arg_meaning.flags.writeable = False
            else:
                arg_meaning = np.array(
                    [atom.func("model-invariant")] * self.N_OF_MODELS
                )
                arg_meaning.flags.writeable = False
        elif atom.content in [True, False]:
            arg_meaning = bitarray.bitarray()
            for model in self.generate_universe():
                atom.append(atom.func(model))
        elif type(atom.content) in [int, float]:
            arg_meaning = np.array()
            for model in self.generate_universe():
                arg_meaning = np.append(arg_meaning, atom.func(model))
        else:  # set case
            arg_meaning = []
            cur_size = 1
            set_repr = np.zeros(
                (cur_size, self.number_of_subsets ** cur_size), dtype=np.uint8
            )  # base case
            offset = 0
            for ind, model in enumerate(self.generate_universe()):
                if ind == self.SIZE2RANGE[cur_size][1]:
                    cur_size += 1
                    offset += set_repr.shape[1]
                    set_repr.flags.writeable = False
                    arg_meaning.append(set_repr)
                    set_repr = np.zeros(
                        (cur_size, self.number_of_subsets ** cur_size),
                        dtype=np.uint8,
                    )
                set_repr[:, ind - offset] = atom.func(model)
                # else:
                #     print(f"now on from {cur_size} to {cur_size + 1}")
                #     # set_repr = np.zeros(
                #     #     (cur_size, 4 ** cur_size), dtype=np.uint8
                #     # )
            else:
                set_repr.flags.writeable = False
                arg_meaning.append(set_repr)
        return arg_meaning

    def get_output_type(self, arg):
        """Get output type of expression, i.e. to what it would evaluate."""
        if type(arg) is tuple:
            return self.operators[arg[0]].output_type
        if type(arg) is float:
            arg = round(arg, 2)
        return self.atom2output_type[str(arg)]

    def compute_meaning(self, expr):
        """Compute meaning representation for given expression"""
        if type(expr) is not tuple:
            return self.output_type2expression2meaning[
                self.atom2output_type[expr]
            ][expr]
        main_operator = self.operators[expr[0]]
        args = expr[1:]
        meanings = []
        for arg in args:
            output_type = self.get_output_type(arg)
            try:
                meanings.append(
                    self.output_type2expression2meaning[output_type][arg]
                )
            except:  # should only occur when evaluating a 'new 'meaning not created in this algorithm
                new_meaning = self.compute_meaning(arg)
                meanings.append(new_meaning)
        return main_operator.func(*meanings)

    def same_meaning(self, el0, el1):
        """Determines if 2 elements have same meaning"""
        if type(el0) != type(el1):
            out = False
        elif type(el0) is list:
            out = all(
                np.array_equal(set_repr0, set_repr1)
                for set_repr0, set_repr1 in zip(el0, el1)
            )
        elif type(el0) is np.ndarray:  # must be bitarray
            out = np.array_equal(el0, el1)
        else:
            pass
        return out

    def _init_atoms(self):
        """Generate all atoms and set certain featues accordingly"""
        # self.expression2meaning = dict()
        self.output_type2expression2meaning = dict(
            (output_type, dict())
            for output_type in [bool, int, float, operators.SetPlaceholder]
        )
        output_type2atoms = {
            int: [],
            float: [],
            bool: [],
            operators.SetPlaceholder: [],
        }
        for i in range(0, self.max_model_size + 1, 2):
            output_type2atoms[int].append(
                Atom(str(i), lambda model, i=i: 0 + i, True)
            )
        set_name2relevant_index = {"A": 0, "B": 1}
        for set_name in ["A", "B"]:
            rel_ind = set_name2relevant_index[set_name]
            output_type2atoms[operators.SetPlaceholder].append(
                Atom(
                    set_name,
                    lambda model, rel_ind=rel_ind: np.array(
                        [int(obj[rel_ind]) for obj in zip(*model)]
                    ),
                    False,
                )
            )
        for q in np.arange(0, 1, 0.1):
            output_type2atoms[float].append(
                Atom(str(round(q, 2)), lambda model, q=q: q, True)
            )
        for boolean in [True, False]:
            output_type2atoms[bool].append(
                Atom(
                    str(boolean), lambda model, boolean=boolean: boolean, True
                )
            )
        self.output_type2atoms = output_type2atoms
        self.atom2output_type = dict()
        for output_type, atoms in output_type2atoms.items():
            for atom in atoms:
                self.atom2output_type[str(atom.content)] = output_type
        self.name2atom = dict(
            (at.content, at)
            for at in it.chain.from_iterable(output_type2atoms.values())
        )
        for at in it.chain.from_iterable(output_type2atoms.values()):
            meaning_atom = self._compute_meaning_atom(at)
            self.output_type2expression2meaning[self.get_output_type(at)][
                at.content
            ] = meaning_atom
            # self.output_type2expression2meaning[atom2output_type][
            #     str(at.content)
            # ] = meaning_atom

    def _ignore_or_add(self, expr):
        """Generates meaning tuple for expression, checks if we already have an
        expression with that meaning, and if so does not do anything with it.
        output_type2expression2meaning mapping according to the output type of
        the expression
        """
        meaning = self.compute_meaning(expr)
        output_type = self.get_output_type(expr)
        if any(
            self.same_meaning(meaning, m)
            for m in self.output_type2expression2meaning[output_type].values()
        ):
            return False
        else:  # unique meaning
            self.output_type2expression2meaning[self.get_output_type(expr)][
                expr
            ] = meaning
            return True

    def generate_all_sentences(self, max_expression_length):
        """Generates all unique meaning sentences up go self.max_quantifier_length and
        self.max_model_size
        :returns: List with all unique sentences
        """
        self._init_atoms()
        self.length2type2expressions = defaultdict(lambda: defaultdict(list))
        for output_type, atoms in self.output_type2atoms.items():
            self.length2type2expressions[1][output_type] = list(
                map(lambda at: at.content, atoms)
            )

        for i in range(2, max_expression_length + 1):
            print(f"Generating new expressions of length {i}")
            type2expressions = defaultdict(list)
            for op_name, operator in self.operators.items():
                output_type = operator.output_type
                input_types = operator.input_types
                if len(input_types) == 1:
                    input_type = operator.input_types[0]
                    for sub_expr in self.length2type2expressions[i - 1][
                        input_type
                    ]:
                        expr = (op_name, sub_expr)
                        unique_meaning = self._ignore_or_add(expr,)
                        if unique_meaning:
                            type2expressions[output_type].append(expr)
                elif len(input_types) == 2:
                    for j in range(1, i - 1):
                        arg_len_1 = j
                        arg_len_2 = i - j - 1
                        arg_type_1 = input_types[0]
                        arg_type_2 = input_types[1]
                        for sub_expr_1 in self.length2type2expressions[
                            arg_len_1
                        ][arg_type_1]:
                            for sub_expr_2 in self.length2type2expressions[
                                arg_len_2
                            ][arg_type_2]:
                                expr = (op_name, sub_expr_1, sub_expr_2)
                                unique_meaning = self._ignore_or_add(expr)
                                if unique_meaning:
                                    type2expressions[output_type].append(expr)
            self.length2type2expressions[i] = type2expressions
            if self.store_at_each_length:
                self.write_expressions()
            new = sum(map(len, type2expressions.values()))
            print(f"{new} unique expressions of length {i} added")
        return list(self.output_type2expression2meaning[bool])

    def unique_meaning(self, exp):
        """Given expression, checks if it has already encountered an expression
        with this meaning"""
        if type(exp) is not tuple:
            return False
        meaning = self.compute_meaning(exp)
        return not any(
            self.same_meaning(meaning, m)
            for m in self.output_type2expression2meaning[
                self.get_output_type(exp)
            ].values()
        )

    def write_expressions(self):
        """Writes current expressions and itself to dill file"""
        print(f"Interim saving of current state (to {self.results_dir}")
        experiment_name = os.path.splitext(os.path.basename(self.json_path))[0]
        current_max_len = max(self.length2type2expressions.keys())
        current_expressions = list(
            it.chain.from_iterable(
                type2expressions[bool]
                for type2expressions in self.length2type2expressions.values()
            )
        )
        with open(
            self.out_dir
            / ("expressions_up_to_length_%s.dill" % current_max_len),
            "wb",
        ) as f:
            dill.dump(current_expressions, f)
        with open(
            self.out_dir
            / ("language_generator_up_to_length_%s.dill" % current_max_len),
            "wb",
        ) as f:
            dill.dump(self, f)

    def exp_with_same_meaning(self, exp):
        """Given expression, returns previously encountered expression with
        same meaning if exists, else returns False"""
        m = self.compute_meaning(exp)
        # if m in self.expression2meaning.values()
        for exp, meaning in self.output_type2expression2meaning[
            self.get_output_type(exp)
        ].items():
            if self.same_meaning(meaning, m):
                return exp
        return False

    def _compute_meaning_matrix(self):
        """returns matrix where the ith row corresponds to the meaning of the
        ith expression (of those it has computed so far)"""
        self.sorted_exps = sorted(
            list(self.output_type2expression2meaning[bool]),
            key=str,
            reverse=True,
        )
        out = np.zeros((len(self.sorted_exps), self.N_OF_MODELS))
        for i, exp in enumerate(self.sorted_exps):
            out[i, :] = self.output_type2expression2meaning[bool][exp]
        self.meaning_matrix = out

    def compute_and_set_meaning_matrix(self):
        """Computes and sets meaning_matrix"""
        if self.meaning_matrix is not None:
            nof_own_expressions = len(
                self.output_type2expression2meaning[bool]
            )
            if self.meaning_matrix.shape[0] == nof_own_expressions:
                return
            else:  # need to make new one
                self._compute_meaning_matrix(self)
