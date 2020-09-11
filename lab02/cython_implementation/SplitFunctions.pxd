import numpy as np
cimport cython
from TypeDefinitions cimport Index, TargetData

cdef class CharacteristicResult:
    cdef double characteristic, \
                left_node_characteristic, \
                right_node_characteristic

cdef class SplitFunctions:
    """
    Non-private functions have interface:

    CharacteristicResult some_characteristic(
        TargetData[:,:] y,
        double node_characteristic,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    args:
        :param y - labels

        :param node_characteristic - characteristic of a
        node-candidate for splitting. It is needed for
        such methods as gain_ratio, information_gain etc

        left_split_indexes - subj

        right_split_indexes - subj

    :return:
        CharacteristicResult
    """
    @staticmethod
    cdef size_t __count_value(
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
    cdef double __gini_impurity(
        TargetData[:, :] y,
        Index[:] observation_indexes
    )

    @staticmethod
    cdef CharacteristicResult gini_index(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef double __entropy(
        TargetData[:,:] y,
        Index[:] observation_indexes
    )

    @staticmethod
    cdef CharacteristicResult information_gain(
        TargetData[:,:] y,
        double entropy_before,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )

    @staticmethod
    cdef CharacteristicResult gain_ratio(
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
    cdef CharacteristicResult mse_sum(
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
    cdef CharacteristicResult mae_sum(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    )
