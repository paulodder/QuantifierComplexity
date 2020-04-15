import pandas as pd
import numpy as np
import networkx as nx
import itertools as it


def signature_quantity(meaning_matrix):
    """Returns a quantity signature vector"""
    out = []
    for model in meaning_matrix.index.values:
        this_signature = [0, 0, 0, 0]
        for a, b in model:
            if a and b:
                this_signature[0] += 1
            elif a:
                this_signature[1] += 1
            elif b:
                this_signature[2] += 1
        out.append(tuple(this_signature))
    return pd.Series(out)


def signature_conservativity(meaning_matrix):
    """Returns a conservativity signature vector"""
    out = []
    for model in meaning_matrix.index:
        idxs = []
        for (a, b) in model:
            if a and not b:
                idxs.append(0)
            if a and b:
                idxs.append(1)
        out.append(tuple(idxs))
    return pd.Series(out)


def is_submodel__types(t0, t1):
    """Given two string representations of models, returns True iff t0 is a
    submodel of t1"""
    return len(t0) == len(t1) and all(
        int(a) or not (int(b)) for a, b in zip(t0, t1)
    )


def is_supermodel__types(t0, t1):
    """Given two string representations of models, returns True iff t0 is a
    submodel of t1"""
    return len(t0) == len(t1) and all(
        int(b) or not (int(a)) for a, b in zip(t0, t1)
    )


def _make_model2submodel_type(meaning_matrix):
    submodel_types = []
    for model in meaning_matrix.index:
        submodel_type = ""
        for (a, b) in model:
            if a == b == True:
                submodel_type += "1"
                # submodel_type.append(1)
            elif a:
                submodel_type += "0"
                # submodel_type.append(0)
        submodel_types.append(submodel_type)
    return pd.Series(index=meaning_matrix.index, data=submodel_types)


def signature_monotonicity(meaning_matrix):
    """Returns a monotonicity meaning matrix"""
    # submodel_types = []
    # Map each model to it's submodeltype (see docs)
    model2submodel_type = _make_model2submodel_type(meaning_matrix)
    # Convert binary matrix to bool to speed computations
    meaning_matrix = meaning_matrix.astype(bool)
    # For each submodel type and expression, compute if there is at least one
    # model satisfying the expression
    # Add column that holds the submodel type for each model (so row)
    meaning_matrix["submodel_type"] = model2submodel_type
    # Create df indexed by submodel types and with columns for each expression,
    # that tells us if a given submodel type has a model on which the
    # expression is true
    submodel_type2has_true_model = meaning_matrix.groupby(
        "submodel_type"
    ).any()
    # Delete the submodel_type column again from the matrix
    del meaning_matrix["submodel_type"]
    # Now that we have this, we the forest to compute for each submodel type,
    # its supermodel types, we then use these to make sure that if a submodel
    # type says true, then its supermodel types will also be true.
    submodel_forest = _make_submodel_forest(
        submodel_type2has_true_model.index.values
    )
    submodel_type2has_true_submodel = pd.DataFrame(
        np.zeros_like(submodel_type2has_true_model.T),
        columns=submodel_type2has_true_model.index.values,
        index=submodel_type2has_true_model.columns,
    )
    for submodel_type in submodel_type2has_true_model.index:
        # Get all submodel types for this submodel type
        submodel_types_relative = list(
            submodel_forest.neighbors(submodel_type)
        )
        # If any of the submodels of this submodel_type satisfy an expression,
        # then our signature value for this expression and any model of this
        # type should be True
        exp2true_in_any = submodel_type2has_true_model.loc[
            submodel_types_relative
        ].any(0)
        # print(exp2true_in_any)
        submodel_type2has_true_submodel[submodel_type] = exp2true_in_any
    submodel_type2has_true_submodel = submodel_type2has_true_submodel.T
    # Now we initialize an empty dataframe where the ith row is indexed by the
    # modeltype of the ith model
    signature_df = pd.DataFrame(
        # index=meaning_matrix.index.map(model2submodel_type).values,
        index=meaning_matrix.index.values,
    )
    # Add a temporary submodel_type column
    signature_df["tmp_submodel_type"] = model2submodel_type
    # We merge based on submodel types
    signature_df = signature_df.merge(
        submodel_type2has_true_submodel,
        left_on="tmp_submodel_type",
        right_index=True,
        how="left",
    )
    del signature_df["tmp_submodel_type"]
    # We reset the index
    signature_df.index = meaning_matrix.index
    return signature_df


def _make_submodel_forest(submodel_types):
    """Produce a graph that represents the submodel relationships the submodel
    types of monotonicity"""
    edges = []
    for t0 in submodel_types:
        for t1 in submodel_types:
            add = True
            if is_submodel__types(t0, t1):
                edges.append((t0, t1))
    g = nx.DiGraph()
    g.add_nodes_from(submodel_types)
    g.add_edges_from(edges)
    return g


def _make_supermodel_forest(submodel_types):
    """Produce a graph that represents the submodel relationships the submodel
    types of monotonicity"""
    edges = []
    for t0 in submodel_types:
        for t1 in submodel_types:
            add = True
            if is_supermodel__types(t0, t1):
                edges.append((t0, t1))
    g = nx.DiGraph()
    g.add_nodes_from(submodel_types)
    g.add_edges_from(edges)
    return g
