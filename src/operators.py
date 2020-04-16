# from SetPlaceholders import SetPlaceholder

import numpy as np
import math
from collections import namedtuple
from itertools import chain

vectorized_isclose = np.vectorize(math.isclose)
vectorized_round = np.vectorize(round)

Operator = namedtuple("Operator", "name func input_types output_type")
SetPlaceholder = namedtuple("Operator", "name func input_types output_type")

Operator.__repr__ = lambda self: self.name
Operator.__str__ = lambda self: self.name

## ff hardcode
SIZE2RANGE = {
    1: (0, 4),
    2: (4, 20),
    3: (20, 84),
    4: (84, 340),
    5: (340, 1364),
    6: (1364, 5460),
    7: (5460, 21844),
    8: (21844, 87380),
    9: (87380, 349524),
    10: (349524, 1398100),
    11: (1398100, 5592404),
    12: (5592404, 22369620),
    13: (22369620, 89478484),
    14: (89478484, 357913940),
    15: (357913940, 1431655764),
    16: (1431655764, 5726623060),
    17: (5726623060, 22906492244),
    18: (22906492244, 91625968980),
    19: (91625968980, 366503875924),
    20: (366503875924, 1466015503700),
}


def index_func(i, set_repr, SIZE2RANGE):
    """Return a set representation of taking the ith index of the given set
    representation"""
    out = []
    for size_minus_one, set_repr_size in enumerate(set_repr):
        size = size_minus_one + 1
        rel_range = SIZE2RANGE[size]
        to_compare = i[rel_range[0] : rel_range[1]]
        result = np.zeros_like(set_repr_size)
        c = set_repr_size.cumsum(0)
        nonzero_x_idxs, nonzero_y_idxs = np.where(c == to_compare)
        x_idxs, mask = np.unique(nonzero_y_idxs, return_index=True)
        result[nonzero_x_idxs[mask], x_idxs] = 1
        out.append(result)
    return out


# def index_func(i, set_repr):
#     global SIZE2RANGE
#     out = []
#     i = i.astype(np.uint8)
#     # print("HERE", i)
#     if all(i == 0):  # no zero based indexing for normal humans
#         return [
#             np.zeros(set_repr_size.shape, dtype=np.uint8)
#             for set_repr_size in set_repr
#         ]
#     for size_minus_one, set_repr_size in enumerate(set_repr):
#         size = size_minus_one + 1
#         rel_range = SIZE2RANGE[size]
#         to_compare = np.repeat(
#             [i[rel_range[0] : rel_range[1]]], set_repr_size.shape[0], axis=0
#         )

#         x, y = np.where(set_repr_size.cumsum(0) == to_compare)
#         new_repr = np.zeros(set_repr_size.shape)
#         new_repr[x, y] = 1
#         out.append(new_repr.astype(np.uint8))
#     return out


def subset_func(set_repr_0, set_repr_1):
    out = np.array([])
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        subset = (
            np.apply_along_axis(max, 0, (set_0 & (1 - set_1))) == 0
        )  # no instances of 0 1
        # print(subset)
        # prevent vacous satisfcation of the subset relation
        at_least_one = np.apply_along_axis(sum, 0, set_0) > 0
        # print(at_least_one)
        out = np.append(out, (subset & at_least_one))
        # out = np.append(out, subset)
    return out.astype(bool)


def diff_func(set_repr_0, set_repr_1):
    out = []
    for size, (set_0, set_1) in enumerate(zip(set_repr_0, set_repr_1)):
        # print(set_0, set_1)
        out.append(set_0 & (1 - set_1))
    # [np.append(out, )
    return out


def mod_func(x, y):
    zero_indices = y <= 0
    y_with_ones = np.array(y[:], copy=True)
    y_with_ones[zero_indices] = 1
    out = x % y_with_ones
    out[zero_indices] = 0
    return out


def div_func(x, y):
    zero_indices = y <= 0
    y_with_ones = np.array(y[:], copy=True)
    y_with_ones[zero_indices] = 1
    out = x / y_with_ones
    out[zero_indices] = 0
    return vectorized_round(out, 2)


def init_operators(max_model_size, number_of_subsets=4):
    cur_cumul = 0
    SIZE2RANGE = dict()
    for size in range(1, max_model_size + 1):
        SIZE2RANGE[size] = (
            sum(number_of_subsets ** s for s in range(1, size)),
            sum(number_of_subsets ** s for s in range(1, size + 1)),
        )
    return {
        "index": Operator(
            "index",
            lambda i, s, size2range=SIZE2RANGE: index_func(i, s, size2range),
            (int, set),
            set,
        ),
        "diff": Operator(
            "diff",
            diff_func,  # (s1.get_set(model) - s2.get_set(model))
            (set, set),
            set,
        ),
        "subset": Operator("subset", subset_func, (set, set), bool),
        ">f": Operator(">f", lambda x, y: x > y, (float, float), bool),
        "=f": Operator(
            "=f", lambda x, y: vectorized_isclose(x, y), (float, float), bool
        ),
        ">": Operator(">", lambda x, y: x > y, (int, int), bool),
        ">=": Operator(">=", lambda x, y: x >= y, (int, int), bool),
        "=": Operator("=", lambda x, y: x == y, (int, int), bool),
        "/": Operator("/", div_func, (int, int), float),
        "-": Operator(
            "-", lambda x, y: x.astype(int) - y.astype(int), (int, int), int
        ),
        "+": Operator(
            "+", lambda x, y: x.astype(int) + y.astype(int), (int, int), int
        ),
        "card": Operator(
            "card",
            lambda set_repr_all: np.array(
                list(
                    chain.from_iterable(
                        np.apply_along_axis(sum, 0, set_repr)
                        for set_repr in set_repr_all
                    )
                )
            ),
            (set,),
            int,
        ),
        "intersection": Operator(
            "intersection",
            lambda x_repr, y_repr: [x & y for x, y in zip(x_repr, y_repr)],
            # lambda model, x, y: tuple(
            #     x.char_func(obj) and y.char_func(obj) for obj in model
            # ),
            (set, set),
            set,
        ),
        "union": Operator(
            "union",
            lambda x_repr, y_repr: [x | y for x, y in zip(x_repr, y_repr)],
            (set, set),
            set,
        ),
        "and": Operator("and", lambda x, y: x & y, (bool, bool), bool),
        "or": Operator("or", lambda x, y: x | y, (bool, bool), bool),
        "not": Operator("not", lambda x: np.invert(x), (bool,), bool),
        "%": Operator("%", mod_func, (int, int), int),
    }


OPERATORS = {
    "index": Operator("index", index_func, (int, set), set),
    "diff": Operator(
        "diff",
        diff_func,  # (s1.get_set(model) - s2.get_set(model))
        (set, set),
        set,
    ),
    "subset": Operator("subset", subset_func, (set, set), bool),
    ">f": Operator(">f", lambda x, y: x > y, (float, float), bool),
    "=f": Operator(
        "=f", lambda x, y: vectorized_isclose(x, y), (float, float), bool
    ),
    ">": Operator(">", lambda x, y: x > y, (int, int), bool),
    ">=": Operator(">=", lambda x, y: x >= y, (int, int), bool),
    "=": Operator("=", lambda x, y: x == y, (int, int), bool),
    "/": Operator("/", div_func, (int, int), float),
    "-": Operator(
        "-", lambda x, y: x.astype(int) - y.astype(int), (int, int), int
    ),
    "+": Operator(
        "+", lambda x, y: x.astype(int) + y.astype(int), (int, int), int
    ),
    "card": Operator(
        "card",
        lambda set_repr_all: np.array(
            list(
                chain.from_iterable(
                    np.apply_along_axis(sum, 0, set_repr)
                    for set_repr in set_repr_all
                )
            )
        ),
        (set,),
        int,
    ),
    "intersection": Operator(
        "intersection",
        lambda x_repr, y_repr: [x & y for x, y in zip(x_repr, y_repr)],
        # lambda model, x, y: tuple(
        #     x.char_func(obj) and y.char_func(obj) for obj in model
        # ),
        (set, set),
        set,
    ),
    "union": Operator(
        "union",
        lambda x_repr, y_repr: [x | y for x, y in zip(x_repr, y_repr)],
        (set, set),
        set,
    ),
    "and": Operator("and", lambda x, y: x & y, (bool, bool), bool),
    "or": Operator("or", lambda x, y: x | y, (bool, bool), bool),
    "not": Operator("not", lambda x: np.invert(x), (bool,), bool),
    # "empty": Operator(
    #     lambda model, x: get_cardinality(model, x) is 0,
    #     (set,),
    #     bool,
    # ),
    # "nonempty": Operator(
    #     lambda model, x: len(x.get_set(model)) > 0, (set,), bool
    # ),
    # "proportion": Operator(
    #     lambda model, X, Y, q: get_cardinality(model, X)
    #     / get_cardinality(model, Y)
    #     > q
    #     if get_cardinality(model, Y) > 0
    #     else 0,
    #     (set, set, float),
    #     bool,
    # )
    # ,
    "%": Operator("%", mod_func, (int, int), int),
}
