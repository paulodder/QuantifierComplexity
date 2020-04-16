import dotenv
import imageio
import sys
import os
import sys
import json

import dill
import bitarray
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from collections import defaultdict


dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
sys.path.insert(0, str(PROJECT_DIR / "src"))
import operators
import utils


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
        experiment_setup="",
        dest_dir=Path(PROJECT_DIR) / Path(RESULTS_DIR_RELATIVE),
        store_at_each_length=1,
    ):
        """Initializes LanguageGenerator object
        :param max_model_size: The size of models on which expressions will be
        evaluated
        :param dest_dir: Directory in which to store results
        :param store_at_each_length: If 1, stores expressions and itself at
        every length
        :param experiment_setup: Name of experiment (corresponds to a .json
        file)"""

        self.max_model_size = max_model_size
        self.dest_dir = dest_dir
        self.store_at_each_length = store_at_each_length
        self.experiment_setup = experiment_setup
        self._load_json(experiment_setup)
        self.N_OF_MODELS = (
            sum(self.number_of_subsets ** i for i in range(max_model_size + 1))
            - 1
        )
        self._set_output_dirs()
        self._set_operators()
        self._set_size2range()
        self.meaning_matrix = None
        self.signature_function2exp2score = dict()

    def _set_output_dirs(self):
        """Set output directories"""
        self.results_dir = Path(
            self.dest_dir
        ) / utils.make_experiment_dir_name(
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

    def _load_json(self, experiment_setup):
        """Loads JSON file from json directory and sets settings accordingly"""
        if experiment_setup:
            with open(
                Path(PROJECT_DIR)
                / Path(JSON_SETUP_DIR_RELATIVE)
                / f"{experiment_setup}.json",
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
            except:  # should only occur when evaluating a 'new 'meaning not
                # created in this algorithm
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
            (output_type, dict()) for output_type in [bool, int, float, set]
        )
        output_type2atoms = {
            int: [],
            # float: [],
            bool: [],
            set: [],
        }
        # Initialize the integer atoms
        for i in range(0, self.max_model_size + 1, 1):
            output_type2atoms[int].append(
                Atom(str(i), lambda model, i=i: i, True)
            )
        set_name2relevant_index = {"A": 0, "B": 1}
        for set_name in ["A", "B"]:
            rel_ind = set_name2relevant_index[set_name]
            output_type2atoms[set].append(
                Atom(
                    set_name,
                    lambda model, rel_ind=rel_ind: np.array(
                        [int(obj[rel_ind]) for obj in zip(*model)]
                    ),
                    False,
                )
            )
        # for q in np.arange(0, 1, 0.1):
        #     output_type2atoms[float].append(
        #         Atom(str(round(q, 2)), lambda model, q=q: q, True)
        #     )
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
        Else updates output_type2expression2meaning mapping according to the output
        type of the expression
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
                        unique_meaning = self._ignore_or_add(expr)
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
        experiment_name = os.path.splitext(
            os.path.basename(self.experiment_setup)
        )[0]
        current_max_len = max(self.length2type2expressions.keys())
        current_expressions = list(
            it.chain.from_iterable(
                type2expressions[bool]
                for type2expressions in self.length2type2expressions.values()
            )
        )
        print(self.results_dir)
        with open(
            self.results_dir
            / ("expressions_up_to_length_%s.dill" % current_max_len),
            "wb",
        ) as f:
            dill.dump(current_expressions, f)
        with open(
            self.results_dir
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
        """Computes matrix where the cell in the ith row and jth column is 1 iff
        the ith model belongs to the jth expression/quantifier
        :returns: None, but sets self.meaning_matrix accordingly"""
        # Sort all expressions that evaluate to a boolean
        sorted_exps = sorted(
            list(self.output_type2expression2meaning[bool]),
            key=str,
            reverse=True,
        )
        out = np.zeros((self.N_OF_MODELS, len(sorted_exps)))
        # For each ith expression, set the ith column to the meaning vector for
        # that expression
        for i, exp in enumerate(sorted_exps):
            out[:, i] = self.output_type2expression2meaning[bool][exp]
        # Convert to df, such that it is indexed by the models, and column names
        # are the expressions
        out = pd.DataFrame(
            out,
            columns=sorted_exps,
            index=[utils.tuple_format(m) for m in self.generate_universe()],
        )
        self.meaning_matrix = out.astype(int)

    def get_meaning_matrix(self):
        if self.meaning_matrix is None:
            self._compute_meaning_matrix()
        return self.meaning_matrix

    def _compute_exp2score(self, signature_function):
        """Given a signature_vector function, computes the score for each
        expression
        :param signature_function: Given a generator for the models, constructs
        a vector where the ith element denotes the signature tuple of the ith
        model, or a matrix of same shape as meaning_matrix"""
        mm = self.get_meaning_matrix()
        # Get signatures for this meaning matrix
        signatures = signature_function(mm)
        # Signature for a model is expression invariant if it is simply a vector
        if len(signatures.shape) == 1:
            # Add signature as a column to allow for groupbys
            mm["signature"] = signatures.values
            # Compute for each signature, the probability of it occurring
            signature2prob = mm["signature"].value_counts(1).sort_index()
            # Compute matrix where the element in posn (i, j) denotes the
            # probability of the jth expression being true in a model given
            # that the model is of the ith signature type
            signature_exp2conditional_prob_Q = (
                mm.groupby("signature").mean().sort_index()
            )
            del mm["signature"]
        else:  # Signature for a model depends on the expression
            signature2prob = signatures.apply(
                lambda s: s.value_counts(normalize=True)
            ).fillna(0)
            all_signatures = sorted(np.unique(signatures.values.ravel()))
            # Start computing for each signature value, the probability of
            # an expression being true on a model given signature
            signature_exp2conditional_prob_Q = pd.DataFrame(index=mm.columns)
            for signature_val in all_signatures:
                mask = signatures == signature_val
                # Compute number of models for each expression with this
                # signature
                exp2nof_signature_occurrences = mask.sum()
                # Compute number of models for each expression with this
                # signature that also satisfy the expression
                exp2nof_true_model_occurrences = (mm & mask).sum()
                # For this signature value, compute for each expression it's
                # probability of being true given this signature value
                exp2cond_prob_Q = (
                    exp2nof_true_model_occurrences
                    / exp2nof_signature_occurrences
                )
                signature_exp2conditional_prob_Q[
                    signature_val
                ] = exp2cond_prob_Q
            # Replace nans with 0
            signature_exp2conditional_prob_Q.fillna(0, inplace=True)
            # Take transpose so the df is indexed by signature values, and
            # columns are the expressions
            signature_exp2conditional_prob_Q = (
                signature_exp2conditional_prob_Q.T
            )
        # Now compute the binary entropy for each P(1_Q|signature)
        signature_exp2cond_entropy = utils.element_wise_binary_entropy(
            signature_exp2conditional_prob_Q
        )
        # For the conditional entropy values for each expression-model value,
        # compute the 'weighted' entropy value by multiplying with the
        # probability of that signature occurring.
        signature_exp2cond_entropy_weighted = signature_exp2cond_entropy.mul(
            signature2prob,
            axis="index"  # Axis=index ensures proper multiplication in the case
            # of model-variant signature values (i.e. signature2prob is vector)
        )
        # Series that maps each expression to its probability of being true
        exp2prob = mm.mean()
        # Map each expression to its entropy (i.e. entropy of the RV 1_Q)
        exp2entropy = utils.element_wise_binary_entropy(exp2prob)
        # Now we compute the 'normalized' score by dividing the conditional
        # entropy by the actual entropy. If this value is close to zero we know
        # that knowing the signature value gives us a lot of information on the
        # truth value of the expression in a given model
        exp2normalized_cond_entropy = (
            signature_exp2cond_entropy_weighted.sum() / exp2entropy
        ).fillna(0)
        # Now we subtract the scores from 1, to make sure that those
        # expressions for which scores are low (i.e. signatures tell us a lot)
        # are mapped to values closer to 1
        exp2score = 1 - exp2normalized_cond_entropy
        self.signature_function2exp2score[signature_function.name] = exp2score

    def get_exp2score(self, signature_function):
        if not signature_function.name in self.signature_function2exp2score:
            self._compute_exp2score(signature_function)
        return self.signature_function2exp2score[signature_function.name]

    def _plot_fig_ax3d(self, funcs):
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = [self.get_exp2score(sign_func) for sign_func in funcs]
        name_x, name_y, name_z = [sign_func.name for sign_func in funcs]
        ax.scatter(x, y, z, alpha=0.3)
        ax.set_xlabel(name_x)
        ax.set_ylabel(name_y)
        ax.set_zlabel(name_z)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        max_exp_len = max(self.length2type2expressions.keys())
        fig.suptitle(
            f"{name_x} vs. {name_y} vs. {name_z} scores for expressions\n (max_model_size={self.max_model_size}, max_expression_length={max_exp_len})"
        ).set_size(10)
        return fig, ax

    def _plot_scatter_3d(self, funcs):
        fig, ax = self._plot_fig_ax3d(funcs)
        base_name = "-vs-".join([sign_func.name for sign_func in funcs])
        fname = f"{base_name}.png"
        fig.savefig(self.figure_dir / fname)
        return self.figure_dir / fname

    def _plot_scatter_2d(self, funcs):
        x, y = [self.get_exp2score(sign_func) for sign_func in funcs]
        max_exp_len = max(self.length2type2expressions.keys())
        name_x, name_y = [sign_func.name for sign_func in funcs]
        plt.close("all")
        sc = plt.scatter(x, y,)
        plt.xlabel(name_x)
        plt.xlim(0, 1)
        plt.ylabel(name_y)
        plt.ylim(0, 1)
        plt.axis("equal")
        plt.title(
            f"{name_x} vs. {name_y} scores for expressions\n (max_model_size={self.max_model_size}, max_expression_length={max_exp_len})"
        ).set_size(10)
        fname = f"{name_x}-vs-{name_y}.png"
        plt.savefig(self.figure_dir / fname)
        return self.figure_dir / fname

    def plot_png(self, funcs):
        """Given a list of 2 or 3 functions, produces a scatter plot accordingly"""
        if len(funcs) == 2:
            out_fname = self._plot_scatter_2d(funcs)
        if len(funcs) == 3:
            out_fname = self._plot_scatter_3d(funcs)
        return out_fname

    def plot_gif(self, funcs, angle_shift_per_frame=30, frame_rate=1 / 24):
        """Given a list of 2 or 3 functions, produces a scatter plot accordingly"""
        tmp_dir = Path(self.figure_dir / "pngs_for_gifs")
        tmp_dir.mkdir(exist_ok=True)
        fig, ax = self._plot_fig_ax3d(funcs)
        base_name = "-".join(f.name for f in funcs)
        images = []
        for angle in range(0, 360, angle_shift_per_frame):
            ax.view_init(30, angle)
            plt.draw()
            out_fname = f"{base_name}_rotation={str(angle).zfill(3)}.png"
            fig.savefig(tmp_dir / out_fname)
            images.append(imageio.imread(tmp_dir / out_fname))
        fname_gif = f"{base_name}-angle_shift={angle_shift_per_frame}-fr={round(frame_rate, 3)}.gif"
        imageio.mimsave(
            self.figure_dir / fname_gif, images, duration=frame_rate,
        )
        return self.figure_dir / fname_gif
