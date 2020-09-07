import numpy as np
cimport cython
from TypeDefinitions cimport Index, TargetData

# TODO move to private some methods
cdef class SplitFunctions:
    """
    Non-private functions have interface:
    double some_charachteristic(
        TargetData[:,:] y,
        double node_charachteristic,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    args:
        y - labels

        node_charachteristic - characteristic of a
        node-candidate for splitting. It is not required for
        some_charachteristic function if variable name is '_'

        left_split_indexes - subj

        right_split_indexes - subj
    """
    @staticmethod
    cdef unsigned long __count_value(
        TargetData[:,:] y,
        Index[:] indexes,
        TargetData value
    )

    @staticmethod
    cdef (double, double) __calc_probabilities(
        TargetData[:,:] y,
        Index[:] observation_indexes
    )

    @staticmethod
    cdef double gini_impurity(
        TargetData[:, :] y,
        Index[:] observation_indexes
    )

    @staticmethod
    cdef double gini_index(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef double entropy(
        TargetData[:,:] y,
        Index[:] observation_indexes
    )

    @staticmethod
    cdef double information_gain(
        TargetData[:,:] y,
        double entropy_before,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef double gain_ratio(
        TargetData[:,:] y,
        double entropy_before,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef double __mean(
        TargetData[:,:] y,
        Index[:] indexes
    )

    @staticmethod
    cdef double __mse(
        TargetData[:,:] y,
        Index[:] indexes
    )

    @staticmethod
    cdef double mse_sum(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef double __mae(
        TargetData[:,:] y,
        Index[:] indexes
    )

    @staticmethod
    cdef double mae_sum(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )
