# distutils: language = c++

from TypeDefinitions cimport TargetData, FeaturesData, Index, NPIndex, AIndex, AFeaturesData
from SplitFunctions cimport SplitFunctions, CharacteristicResult
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.limits cimport numeric_limits
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort

cimport numpy as np
import numpy as np

from cpython cimport array
import array

# TODO remove alloc/dealloc
cdef class Node:
    cdef size_t feature
    cdef double split_value

    cdef Node left, right

    # saving it for pruning
    cdef array.array observation_indexes

    cdef double answer

    cdef __init__(
        self,
        size_t feature,
        double split_value,
        array.array observation_indexes,
        Node left=None,
        Node right=None,
        double answer=0
    ):
        self.feature = feature
        self.split_value = split_value
        self.observation_indexes = observation_indexes
        self.left = left
        self.right = right
        self.answer = answer

    cdef is_leaf(self):
        return not self.left and not self.right

# map criterion name to criterion pointer
cdef unordered_map[
    string,
    double(*)(
        TargetData[:],
        double,
        Index[:],
        Index[:]
    )
] criterion_name_to_calculator_method

criterion_name_to_calculator_method[b'gini'] = SplitFunctions.gini_index
criterion_name_to_calculator_method[b'entropy'] = SplitFunctions.information_gain
criterion_name_to_calculator_method[b'gain_ratio'] = SplitFunctions.gain_ratio
criterion_name_to_calculator_method[b'mse'] = SplitFunctions.mse_sum
criterion_name_to_calculator_method[b'mae'] = SplitFunctions.mae_sum

# define comparators
cdef cmp_lesser_is_better(double x, double y):
    return x - y

cdef cmp_bigger_is_better(double x, double y):
    return y - x

# map criterion name to criterion comparator
cdef unordered_map[
    string,
    double(*)(double, double)
] criterion_name_to_cmp

criterion_name_to_cmp[b'gini'] = cmp_lesser_is_better
criterion_name_to_cmp[b'entropy'] = cmp_bigger_is_better
criterion_name_to_cmp[b'gain_ratio'] = cmp_bigger_is_better
criterion_name_to_cmp[b'mse'] = cmp_bigger_is_better
criterion_name_to_cmp[b'mae'] = cmp_bigger_is_better

cdef class FeatureBestSplit:
    cdef size_t split_feature

    cdef CharacteristicResult characteristics
    cdef double split_value

    cdef bint empty

    def __init__(
        self,
        size_t feature,
        CharacteristicResult characteristics,
        double split_value,
        empty
    ):
        self.split_feature = feature

        self.characteristics = characteristics
        self.split_value = split_value
        self.empty = empty

cdef cmp_FeatureBestSplit(const FeatureBestSplit &a, const FeatureBestSplit &b):
    if a.characteristics.characteristic == b.characteristics.characteristic:
        return a.split_feature < b.split_feature  # use feature as id
    return a.characteristics.characteristic < b.characteristics.characteristic

# CART DT (maybe)
cdef class DecisionTree:
    cdef double(*cmp_criterion_values)(double, double)
    cdef double(*calc_criterion_value)(
        TargetData[:],
        double,
        Index[:],
        Index[:]
    )

    cdef str task, \
             splitter

    cdef size_t max_depth, \
                min_samples_split, \
                max_features

    cdef np.RandomState random_state

    cdef Node root

    cdef FeaturesData[:,:] X
    cdef TargetData[:,:] y

    def __init__(
        self,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        max_features,
        random_state=42
    ):
        self.cmp_criterion_values = \
            criterion_name_to_cmp[criterion.encode()]
        self.calc_criterion_value = \
            criterion_name_to_calculator_method[criterion.encode()]

        if criterion in ['mse', 'mae']:
            self.task = 'regression'
        else:
            self.task = 'classification'

        self.splitter = splitter

        if self.max_depth is None:
            self.max_depth = numeric_limits[size_t].max()
        else:
            self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.max_features = max_features

        self.random_state = random_state

        self.root = None

    cdef fit(
        self,
        FeaturesData[:,:] X,
        TargetData[:,:] y
    ):
        self.X = X
        self.y = y

        self.__construct_tree()

    cdef predict(self, FeaturesData[:,:] X):
        cdef list predictions = []

        cdef size_t i

        for i in range(X.shape[0]):
            cdef double prediction = self.__predict_observation(X[i, :])
            predictions.append(prediction)

        return np.array(predictions).reshape((-1, 1))

    cdef double __predict_observation(self, FeaturesData[:] el):
        cdef Node node = self.root

        while not node.is_leaf():
            if el[node.feature] <= node.split_value:
                node = node.left
            else:
                node = node.right

        return node.answer

    cdef (array.array, array.array, array.array) __get_split_pairs(
        self,
        size_t feature,
        Index[:] observation_indexes
    ):
        """
        :param feature: feature used to split node observations
        :param observation_indexes: subj
        
        :return: ordered_observation_indexes, uniques, split_indexes
            ordered_observation_indexes - observations_indexes ordered by
                feature values
            
            uniques - stores unique values of current feature
            
            split_indexes - stores indexes i: \\forall j in [0; i)
                self.X[ordered_observation_indexes[j]] <= uniques[k],
                \\forall j in [i, observations_size)
                self.X[ordered_observation_indexes[j]] > uniques[k]
                where split_indexes[k] = i
        """
        # node observations here is equal to observations
        # in order to simplify code
        cdef size_t observations_size = observation_indexes.shape[0]

        cdef vector[pair[FeaturesData, Index]] indexed_observations(
            observations_size
        )

        cdef size_t i

        for i in range(observations_size):
            indexed_observations[i].first = self.X[
                observation_indexes[i], feature
            ]
            indexed_observations[i].second = observation_indexes[i]

        sort(indexed_observations.begin(), indexed_observations.end())

        cdef array.array ordered_observation_indexes = array.array(AIndex)

        for i in range(observations_size):
            ordered_observation_indexes.append(
                indexed_observations[i].second
            )

        cdef array.array uniques = array.array(AFeaturesData), \
                         split_indexes = array.array(AIndex)

        # loop variables
        cdef double current_unique
        # index of first not equal to current_unique element
        cdef size_t last_i = 0

        while True:
            current_unique = indexed_observations[last_i].first

            for i in range(last_i + 1, observations_size):
                if indexed_observations[i].first != current_unique:
                    last_i = i
                    break

            # if there are no different values left
            if last_i != i:
                break

            uniques.append(current_unique)
            split_indexes.append(last_i)

        return ordered_observation_indexes, uniques, split_indexes

    cdef FeatureBestSplit __find_best_split_by_feature(
        self,
        size_t feature,
        Index[:] observation_indexes,
        double node_characteristic
    ):
        cdef array.array ordered_observation_indexes, \
                         uniques, \
                         split_indexes

        ordered_observation_indexes, uniques, split_indexes = \
            self.__get_split_pairs(
                feature,
                observation_indexes
            )

        cdef Index[:] ordered_observation_indexes_view = \
            ordered_observation_indexes

        # find best split
        cdef CharacteristicResult best_result
        cdef bint best_result_initialised = False
        cdef size_t best_unique_index  # splitting unique
        cdef size_t best_split_index

        # loop variables
        cdef size_t i, \
                    split_index
        cdef CharacteristicResult current_result

        for i in range(len(uniques)):
            split_index = split_indexes[i]

            current_result = \
                self.calc_criterion_value(
                    self.y,
                    node_characteristic,
                    ordered_observation_indexes_view[:split_index],
                    ordered_observation_indexes_view[split_index:]
                )

            if not best_result_initialised or \
               self.cmp_criterion_values(
                   current_result.characteristic,
                   best_result.characteristic
               ) < 0:
                best_result = current_result
                best_unique_index = i
                best_split_index = split_index
                best_result_initialised = True

        cdef FeatureBestSplit result = FeatureBestSplit(
            feature,
            best_result,
            uniques[best_unique_index],
            not best_result_initialised
        )

        return result

    cdef (array.array, array.array) __get_splitted_indexes(
        self,
        Index[:] observation_indexes,
        size_t split_feature,
        double split_value
    ):
        cdef array.array left_split_indexes  = array.array(AIndex), \
                         right_split_indexes = array.array(AIndex)

        # loop variables
        cdef size_t i
        cdef size_t current_index

        for i in range(observation_indexes.shape[0]):
            current_index = observation_indexes[i]

            if self.X[current_index, split_feature] <= split_value:
                left_split_indexes.append(current_index)
            else:
                right_split_indexes.append(current_index)

        return left_split_indexes, right_split_indexes

    cdef Node __construct_tree_helper(
        self,
        Index[:] observation_indexes,
        double node_characteristic,
        size_t depth
    ):
        cdef Node not_splitted_node = Node(
            0,
            0,
            np.asarray(observation_indexes),
            answer=self.__get_answer(
                observation_indexes
            )
        )

        if depth > self.max_depth or \
           observation_indexes.shape[0] < self.min_samples_split:
            return not_splitted_node

        cdef np.ndarray[np.uint64_t] features = np.arange(
            self.X.shape[1], dtype=np.uint64
        )

        if self.splitter == 'random':
            self.random_state.shuffle(features)

        cdef FeatureBestSplit best_feature_split
        cdef bint best_feature_split_initialised = False

        cdef size_t feature_counter = 0

        # loop variables
        cdef size_t feature
        cdef FeatureBestSplit feature_split

        for feature in features:
            if best_feature_split_initialised and \
               self.splitter == 'random' and \
               feature_counter > self.max_features:
                break

            feature_split = self.__find_best_split_by_feature(
                feature,
                observation_indexes
            )

            if feature_split.empty:
                continue

            if not best_feature_split_initialised or \
               self.cmp_criterion_values(
                   feature_split.characteristics.characteristic,
                   best_feature_split.characteristics.characteristic
               ) < 0:
                best_feature_split = feature_split
                best_feature_split_initialised = True

            feature_counter += 1

        if not best_feature_split_initialised or \
           best_feature_split.characteristics.characteristic == 0:
            return not_splitted_node

        node = Node(
            best_feature_split.split_feature,
            best_feature_split.split_value,
            observation_indexes
        )

        cdef array.array left_split_indexes, right_split_indexes

        left_split_indexes, right_split_indexes = \
            self.__get_splitted_indexes(
                observation_indexes,
                best_feature_split.split_feature,
                best_feature_split.split_value
            )

        node.left = self.__construct_tree_helper(
            left_split_indexes,
            best_feature_split.characteristics.left_node_characteristic,
            depth + 1
        )

        node.right = self.__construct_tree_helper(
            right_split_indexes,
            best_feature_split.characteristics.right_node_characteristic,
            depth + 1
        )

        return node

    cdef double __get_answer(self, Index[:] observation_indexes):
        if self.task == 'classification':
            cdef size_t ones = SplitFunctions.__count_value(
                self.y,
                observation_indexes,
                1
            )

            cdef double threshold = observation_indexes.shape[0] / 2

            if ones > threshold:
                return 1
            elif ones < threshold:
                return 0
            else:
                return self.random_state.randint(0, 2)
        else:
            return SplitFunctions.__mean(
                self.y,
                observation_indexes
            )

    cdef __construct_tree(self):
        cdef array.array observation_indexes = \
            array.array(AIndex, range(self.y.shape[0]))

        # TODO init node_characteristic
        self.root = self.__construct_tree_helper(
            observation_indexes,
            node_characteristic=1,
            depth=1
        )