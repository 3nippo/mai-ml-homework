cdef class CharacteristicResult:
    def __init__(
        self,
        double characteristic,
        double left_node_characteristic,
        double right_node_characteristic
    ):
        self.characteristic = characteristic
        self.left_node_characteristic = left_node_characteristic
        self.right_node_characteristic = right_node_characteristic

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
    ):
        cdef size_t counter = 0
        cdef size_t i

        for i in range(indexes.shape[0]):
            counter += y[indexes[i], 0] == value

        return counter

    @staticmethod
    cdef (double, double) __calc_probabilities(
        TargetData[:,:] y,
        Index[:] observation_indexes
    ):
        cdef long observation_num = observation_indexes.shape[0]

        cdef double p_0 = SplitFunctions.__count_value(
            y,
            observation_indexes,
            0
        )
        p_0 /= observation_num

        cdef double p_1 = SplitFunctions.__count_value(
            y,
            observation_indexes,
            1
        )
        p_1 /= observation_num

        return p_0, p_1

    @staticmethod
    cdef double __gini_impurity(
        TargetData[:, :] y,
        Index[:] observation_indexes
    ):
        cdef double p_0, p_1

        p_0, p_1 = SplitFunctions.__calc_probabilities(
            y,
            observation_indexes
        )

        return 1 - p_0 * p_0 - p_1 * p_1
    
    @staticmethod
    cdef CharacteristicResult gini_index(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    ):
        cdef size_t w_left = left_split_indexes.shape[0]
        cdef size_t w_right = right_split_indexes.shape[0]

        cdef size_t W = w_left + w_right

        cdef double gini_index_left = w_left/W * SplitFunctions.__gini_impurity(
            y,
            left_split_indexes
        )

        cdef double gini_index_right = w_right/W * SplitFunctions.__gini_impurity(
            y,
            right_split_indexes
        )

        cdef CharacteristicResult result = CharacteristicResult(
            gini_index_left + gini_index_right,
            0,
            0
        )

        return result
    
    @staticmethod
    cdef double __entropy(
        TargetData[:,:] y,
        Index[:] observation_indexes
    ):
        cdef double p_0, p_1

        p_0, p_1 = SplitFunctions.__calc_probabilities(
            y,
            observation_indexes
        )

        cdef double entropy = 0

        if p_0 != 0:
            entropy -= p_0*np.log2(p_0)

        if p_1 != 0:
            entropy -= p_1*np.log2(p_1)

        return entropy

    @staticmethod
    cdef CharacteristicResult information_gain(
        TargetData[:,:] y,
        double entropy_before,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    ):
        cdef double left_node_entropy = SplitFunctions.__entropy(
            y,
            left_split_indexes
        )

        cdef double right_node_entropy = SplitFunctions.__entropy(
            y,
            right_split_indexes
        )

        cdef CharacteristicResult result = CharacteristicResult(
            entropy_before - left_node_entropy - right_node_entropy,
            left_node_entropy,
            right_node_entropy
        )

        return result

    @staticmethod
    cdef CharacteristicResult gain_ratio(
        TargetData[:,:] y,
        double entropy_before,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    ):
        cdef double split_info = 0

        cdef size_t w_left = left_split_indexes.shape[0]
        if w_left != 0:
            split_info += w_left * np.log2(w_left)

        cdef size_t w_right = right_split_indexes.shape[0]
        if w_right != 0:
            split_info += w_right * np.log2(w_right)

        cdef double information_gain, \
                    left_node_entropy, \
                    right_node_entropy

        information_gain, left_node_entropy, right_node_entropy = \
            SplitFunctions.information_gain(
                y,
                entropy_before,
                left_split_indexes,
                right_split_indexes
            )

        cdef CharacteristicResult result = CharacteristicResult(
            information_gain / split_info,
            left_node_entropy,
            right_node_entropy
        )

        return result

    @staticmethod
    cdef double __mean(
        TargetData[:,:] y,
        Index[:] indexes
    ):
        cdef size_t i
        cdef double mean = 0

        for i in range(indexes.shape[0]):
            mean += y[indexes[i], 0] / indexes.shape[0]

        return mean

    @staticmethod
    cdef double __mse(
        TargetData[:,:] y,
        Index[:] indexes
    ):
        cdef size_t i
        cdef double mse = 0, \
                    mean = SplitFunctions.__mse(y, indexes)

        for i in range(indexes.shape[0]):
            mse += (y[indexes[i], 0] - mean) \
                   * (y[indexes[i], 0] - mean) \
                   / indexes.shape[0]

        return mse

    @staticmethod
    cdef CharacteristicResult mse_sum(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    ):
        cdef double left_mse = SplitFunctions.__mse(y, left_split_indexes), \
                    right_mse = SplitFunctions.__mse(y, right_split_indexes)

        cdef CharacteristicResult result = CharacteristicResult(
            left_mse + right_mse, 0, 0
        )

        return result

    @staticmethod
    cdef double __mae(
        TargetData[:,:] y,
        Index[:] indexes
    ):
        cdef double mae = 0, \
                    mean = SplitFunctions.__mean(y, indexes)

        cdef unsigned int i

        for i in range(indexes.shape[0]):
            mae += abs(y[indexes[i], 0] - mean) / indexes.shape[0]

        return mae

    @staticmethod
    cdef CharacteristicResult mae_sum(
        TargetData[:,:] y,
        double _,
        Index[:] left_split_indexes,
        Index[:] right_split_indexes
    ):
        cdef double left_mae = SplitFunctions.__mae(y, left_split_indexes), \
                    right_mae = SplitFunctions.__mae(y, right_split_indexes)

        cdef CharacteristicResult result = CharacteristicResult(
            left_mae + right_mae, 0, 0
        )

        return result
