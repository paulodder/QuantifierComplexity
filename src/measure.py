import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import math
import dill
import numpy as np
from collections import Counter, defaultdict
import imp
import dotenv
import os
import importlib

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_DIR = os.getenv("PROJECT_DIR")
JSON_SETUP_DIR_RELATIVE = os.getenv("JSON_SETUP_DIR_RELATIVE")
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
sys.path.insert(0, PROJECT_DIR)
from src import utils

importlib.reload(utils)

MAX_EXPRESSION_LENGTH = 12
MAX_MODEL_SIZE = 8
import sys

sys.argv = [""]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_expression_length",
        "-l",
        type=int,
        default=MAX_EXPRESSION_LENGTH,
        help="expression length",
    )
    parser.add_argument(
        "--max_model_size",
        "-m",
        type=int,
        default=MAX_MODEL_SIZE,
        help="model size",
    )
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="Logical_index",
        help="Name of experiment",
    )
    parser.add_argument(
        "--comparison_experiment_name",
        "-c",
        type=str,
        default=None,
        help="Name of experiment to compare with",
    )
    # parser.add_argument(
    #     "--number_of_subsets",
    #     "-s",
    #     type=int,
    #     default=DEFAULT_NOF_SUBSETS,
    #     help="number of subsets to use include (if 4 uses all, if 3 uses just just A and B",
    # )

    # parser.add_argument(
    #     "--dest_dir",
    #     "-d",
    #     type=str,
    #     default=DEFAULT_DEST_DIR,
    #     help="Dir to write results to",
    # )
    # parser.add_argument(
    #     "--sesref_prefix",
    #     "-p",
    #     type=str,
    #     default="ALL",
    #     help="prefix of sessionreferences to be included",
    # )
    return parser.parse_args()


# def quant_for_exps(exps, lang_gen):
#     meaning_matrix = return_meaning_matrix(exps, lang_gen)
#     for i, exp in enumerate(exps):
#         outp_type = lang_gen.get_output_type(exp)
#         meaning_matrix[i, :] = lang_gen.output_type2expression2meaning[
#             outp_type
#         ][exp]
#     H_Q_given_char_tuple = np.zeros(len(exps), dtype=np.float128)
#     H_Q = vectorized_h(
#         np.array(
#             (np.apply_along_axis(sum, 1, meaning_matrix) / nof_models),
#             dtype=np.float128,
#         )
#     )
#     for char_tuple, prob in char_tuple2freq.items():
#         # relevant_bools = meaning_matrix[char_tuple2bool_idxs[char_tuple]]
#         print(char_tuple)
#         relevant_sample_space = meaning_matrix[
#             :, char_tuple2bool_idxs[char_tuple]
#         ]
#         size_of_sample_space = relevant_sample_space.shape[1]
#         # print(size_of_sample_space)
#         freqs = (
#             np.apply_along_axis(sum, 1, relevant_sample_space)
#             / size_of_sample_space
#         )
#         H_Q_given_this_char_tuple = vectorized_h(freqs) * prob
#         print(H_Q_given_this_char_tuple)
#         H_Q_given_char_tuple += H_Q_given_this_char_tuple
#         # (entropy_of_Q - H_Q_given_char_tuple)
#     exp2H_Q = dict((e, q) for e, q in zip(exps, H_Q))
#     exp2H_Q_given_char_tuple = dict(
#         (e, q) for e, q in zip(exps, H_Q_given_char_tuple)
#    n )
#     # return exp2H_Q, exp2H_Q_given_char_tuple  # H_Q, H_Q_given_char_tuple
#     return dict(
#         (exp, q)
#         for exp, q in zip(
#             exps,
#             [
#                 ((b - a) / b) if b != 0 else 1
#                 for a, b in zip(H_Q_given_char_tuple, H_Q)
#             ],
#         )
#     )


# return meaning_matrix
#     [np.array(lang_gen.compute_meaning(exp)) for exp in exps]
# )
# Q_distr = {
#     1: sum(meaning) / len(meaning),
#     0: 1 - (sum(meaning) / len(meaning)),
# }
# entropy_of_Q = h(Q_distr[1])
# # Q_given_char_tuple2prob = compute_conditional_entropy(char_tuple2bool_idxs, Q_distr)
# H_Q_given_char_tuple = 0
# for char_tuple, prob in char_tuple2freq.items():
#     # now compute entropy of Q given this char_tuple
#     relevant_bools = meaning[char_tuple2bool_idxs[char_tuple]]
#     freqs = sum(relevant_bools) / len(relevant_bools)
#     H_Q_given_this_char_tuple = char_tuple2freq[char_tuple] * h(freqs)
#     H_Q_given_char_tuple += H_Q_given_this_char_tuple
#     # print(
#     #     "Knowing %s means entropy of %s, all %s"
#     #     % (char_tuple, H_Q_given_char_tuple, freqs)
#     # )
# return (
#     (entropy_of_Q - H_Q_given_char_tuple) / entropy_of_Q
#     if entropy_of_Q != 0
#     else 1
# )


# def quant(exp, lang_gen):
#     meaning = lang_gen.compute_meaning(exp)
#     Q_distr = {
#         1: sum(meaning) / len(meaning),
#         0: 1 - (sum(meaning) / len(meaning)),
#     }
#     entropy_of_Q = h(Q_distr[1])
#     # Q_given_char_tuple2prob = compute_conditional_entropy(char_tuple2bool_idxs, Q_distr)
#     H_Q_given_char_tuple = 0
#     for char_tuple, prob in char_tuple2freq.items():
#         # now compute entropy of Q given this char_tuple
#         relevant_bools = meaning[char_tuple2bool_idxs[char_tuple]]
#         freqs = sum(relevant_bools) / len(relevant_bools)
#         H_Q_given_this_char_tuple = char_tuple2freq[char_tuple] * h(freqs)
#         H_Q_given_char_tuple += H_Q_given_this_char_tuple
#         # print(
#         #     "Knowing %s means entropy of %s, all %s"
#         #     % (char_tuple, H_Q_given_char_tuple, freqs)
#         # )
#     # return entropy_of_Q, H_Q_given_char_tuple
#     return (
#         (entropy_of_Q - H_Q_given_char_tuple) / entropy_of_Q
#         if entropy_of_Q != 0
#         else 1
#     )


# (
#         (entropy_of_Q - H_Q_given_char_tuple) / entropy_of_Q
#         if entropy_of_Q != 0
#         else 1
#     )
# for val, P_q in Q_distr.items():
#     for


if __name__ == "__main__":
    args = parse_args()
    exps = sorted(
        utils.load_expressions_for(args.experiment_name, args.max_model_size),
        key=str,
        reverse=True,
    )
    lang_gen = utils.load_lang_gen_for(
        args.experiment_name, args.max_model_size
    )
    # char_matrix = utils.compute_char_tuple_matrix(lang_gen)
    # (char_tuple2freq, char_tuple2bool_idxs,) = utils.init_char_tuple_distribution(
    #     char_matrix, lang_gen
    # )
    meaning_matrix = utils.return_meaning_matrix(exps, lang_gen)
    points_to_plot = []
    plt.close("all")
    figure_dir = Path(
        Path(PROJECT_DIR)
        / RESULTS_DIR_RELATIVE
        / Path(
            utils.make_experiment_dir_name(
                args.max_model_size, args.experiment_name
            )
        )
        / "figures"
    )
    os.mkdir(figure_dir) if not os.path.exists(figure_dir) else None
    exp2quant = utils.quant_for_exps(
        exps,
        lang_gen,
        meaning_matrix,  # , char_tuple2freq, char_tuple2bool_idxs
    )
    utils.plot_score_distribution_by_length_scatter(
        exp2quant, "Quantity", args.max_model_size, figure_dir
    )
    utils.plot_score_distribution_by_length_histo(
        exp2quant, "Quantity", args.max_model_size, figure_dir
    )

    exp2monotonicity = utils.new_monotonicity_for_exps(
        exps,
        lang_gen,
        meaning_matrix,  # , char_tuple2freq, char_tuple2bool_idxs
    )
    # exp2monotonicity = dict((e, v) for e, v in zip(exps, exp2monotonicity))
    utils.plot_score_distribution_by_length_histo(
        exp2monotonicity, "Monotonicity", args.max_model_size, figure_dir
    )

    utils.plot_score_distribution_by_length_scatter(
        exp2monotonicity, "Monotonicity", args.max_model_size, figure_dir
    )

    exp2cons = utils.conservativity_for_exps(
        exps,
        lang_gen,
        meaning_matrix,  # , char_tuple2freq, char_tuple2bool_idxs
    )
    # exp2cons = dict((e, v) for e, v in zip(exps, exp2cons))
    utils.plot_score_distribution_by_length_scatter(
        exp2cons, "Conservativity", args.max_model_size, figure_dir
    )
    utils.plot_score_distribution_by_length_histo(
        exp2cons, "Conservativity", args.max_model_size, figure_dir
    )
    for r in range(0, 360, 10):
        utils.scatter(
            exp2cons,
            exp2monotonicity,
            exp2quant,
            ["conservativity", "monotonicity", "quantity"],
            args.max_model_size,
            figure_dir,
            rotation=r,
        )
# utils.scatter(
# n    exp2cons,
#     exp2monotonicity,
#     exp2quant,
#     ["conservativity", "monotonicity", "quantity"],
#     args.max_model_size,
#     figure_dir,
#     rotation=30,
# )
# utils.scatter(
#     exp2cons,
#     exp2monotonicity,
#     exp2quant,
#     ["conservativity", "monotonicity", "quantity"],
#     args.max_model_size,
#     figure_dir,
#     rotation=60,
# )
# utils.scatter(
#     exp2cons,
#     exp2monotonicity,
#     exp2quant,
#     ["conservativity", "monotonicity", "quantity"],
#     args.max_model_size,
#     figure_dir,
#     rotation=90,
# )
# utils.scatter(
#     exp2cons,
#     exp2monotonicity,
#     exp2quant,
#     ["conservativity", "monotonicity", "quantity"],
#     args.max_model_size,
#     figure_dir,
#     rotation=120,
# )
# utils.scatter(
#     exp2cons,
#     exp2monotonicity,
#     exp2quant,
#     ["conservativity", "monotonicity", "quantity"],
#     args.max_model_size,
#     figure_dir,
#     rotation=150,
# )

# a, b, c, d, e = utils.monotonicity_for_exps(
#     exps, lang_gen, meaning_matrix  # , char_tuple2freq, char_tuple2bool_idxs
# )


# x, y = [], []
# for exp, q in exp2score.items():
#     x.append(utils.exp_len(exp))
#     y.append(q)
# plt.scatter(x, y, alpha=0.05)
# plt.xlabel("length")
# plt.ylabel(f"{score.lower()} score")
# plt.title(f"{score} score distribution by expression length")
# plt.savefig(f"figures/length-vs-{score.lower()}-score_model_size=9.png")
plt.close("all")

# # let's get expressions for previous
# with open(
#     "/home/paul/Dropbox/courses/quantifiers/SimInf_Quantifiers/MY_RESULTS/Logical_length=8_size=9/expressions_up_to_length_8.dill",
#     "rb",
# ) as f:
#     no_index_expressions = dill.load(f)

# length2no_index_expr = utils.partition_on_length(no_index_expressions)

# number_of_index_formulas = []
# number_of_no_index_formulas = []
# length_range = range(1, args.max_model_size)
# for length in length_range:
#     number_of_index_formulas.append(len(length2exps[length]))
#     number_of_no_index_formulas.append(len(length2no_index_expr[length]))
# plt.close("all")
# plt.plot(length_range, number_of_index_formulas)
# plt.plot(length_range, number_of_no_index_formulas)
# plt.legend(["index in language", "no index"], loc="upper left")
# plt.title(
#     f"Number of unique expressions modulo equivalence for model size={args.max_model_size}"
# )
# plt.savefig("figures/nof_expressions_index-vs-no-index.png")

# # x = []
# # y = []
# # print("HERE")
# # for length, exps in length2exps.items():
# #     print("length", length)
# #     for i, exp in enumerate(exps):
# #         if i % 10 == 0:
# #             print(i)
# #         x.append(length)
# #         y.append(quant(exp, lang_gen))
