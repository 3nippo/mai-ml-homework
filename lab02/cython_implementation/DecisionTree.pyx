# distutils: language = c++

from TypeDefinitions cimport TargetData, FeaturesData, Index, NPIndex
from SplitFunctions cimport SplitFunctions
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
    cdef unsigned long feature
    cdef double split_value

    cdef Node *left, *right

    # saving it for pruning
    cdef np.ndarray[NPIndex] observation_indexes

    cdef double answer

    cdef __init__(
        self,
        unsigned long feature,
        double split_value,
        np.ndarray[NPIndex] observation_indexes,
        Node *left=None,
        Node *right=None,
        double answer=0
    ):
        self.feature = feature
        self.split_value = split_value
        self.observation_indexes = observation_indexes
        self.left = left
        self.right = right
        self.answer = answer

    cdef __dealloc__(self):
        if self.left != NULL:
            PyMem_Free(self.left)
        if self.right != NULL:
            PyMem_Free(self.right)

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

    cdef unsigned long max_depth, \
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
            self.max_depth = numeric_limits[ulong].max()
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

        cdef unsigned long i

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

    def __get_split_pairs_gen(
        self,
        unsigned long feature,
        Index[:] observation_indexes
    ):
        # node observations here is equal to observations
        # in order to simplify code
        cdef unsigned long observations_size = observation_indexes.shape[0]

        cdef vector[pair[FeaturesData, Index]] indexed_observations(
            observations_size
        )

        cdef unsigned long i

        for i in range(observations_size):
            indexed_observations[i].first = self.X[
                observation_indexes[i], feature
            ]
            indexed_observations[i].second = observation_indexes[i]

        sort(indexed_observations.begin(), indexed_observations.end())

        cdef array.array ordered_observation_indexes = array.array('Q')

        for i in range(observations_size):
            ordered_observation_indexes.append(
                indexed_observations[i].second
            )

        cdef unsigned long last_i = 0

        # loop variables
        cdef double current_unique
        cdef Index[:] left_split_indexes, right_split_indexes

        while True:
            current_unique = indexed_observations[last_i].first

            for i in range(last_i + 1, observations_size):
                if indexed_observations[i].first != current_unique:
                    last_i = i
                    break

            # if there are no different values left
            if last_i != i:
                break

            left_split_indexes = ordered_observation_indexes[:last_i]
            right_split_indexes = ordered_observation_indexes[last_i:]

            yield left_split_indexes, right_split_indexes, current_unique

        return ordered_observation_indexes

    # TODO start here
    BestFeatureSplit = namedtuple(
        'BestFeatureSplit',
        [
            'criterion_value',
            'split_value',
            'left_split_indexes',
            'right_split_indexes'
        ]
    )

    # output can be (None, None)
    def __find_best_split_by_feature(
        self,
        feature,
        observation_indexes
    ):
        split_pairs_gen = self.__get_split_pairs_gen(
            feature,
            observation_indexes
        )

        # find best split
        best_split = DecisionTree.BestFeatureSplit(
            None,
            None,
            None,
            None
        )

        for left_split_indexes, right_split_indexes, split_value in split_pairs_gen:
            criterion_value = self.__calc_criterion_value(
                self,
                observation_indexes,
                left_split_indexes,
                right_split_indexes
            )

            current_split = DecisionTree.BestFeatureSplit(
                criterion_value,
                split_value,
                left_split_indexes,
                right_split_indexes
            )

            if best_split.criterion_value is None or \
               self.__cmp_criterion_values(
                   criterion_value,
                   best_split.criterion_value
               ) < 0:
                best_split = current_split

        return best_split

    def __construct_tree_helper(self, observation_indexes, depth):
        not_splitted_node = DecisionTree.Node(
            None,
            None,
            observation_indexes,
            answer=self.__get_answer(
                observation_indexes
            )
        )

        if depth > self.max_depth or \
           observation_indexes.shape[0] < self.min_samples_split:
            return not_splitted_node

        features = np.arange(self.X.shape[1])
        if self.splitter == 'random':
            self.random_state.shuffle(features)

        best_feature_split = None
        best_feature = None

        feature_counter = 0

        for feature in features:
            if best_feature and \
               self.splitter == 'random' and \
               feature_counter > self.max_features:
                break

            feature_split = self.__find_best_split_by_feature(
                feature,
                observation_indexes
            )

            if feature_split.criterion_value is None:
                continue

            if best_feature is None or \
               self.__cmp_criterion_values(
                   feature_split.criterion_value,
                   best_feature_split.criterion_value
               ) < 0:
                best_feature_split = feature_split
                best_feature = feature

            feature_counter += 1

        if best_feature is None or \
           best_feature_split.criterion_value == 0:
            return not_splitted_node

        node = DecisionTree.Node(
            best_feature,
            best_feature_split.split_value,
            observation_indexes
        )

        node.left = self.__construct_tree_helper(
            best_feature_split.left_split_indexes,
            depth + 1
        )

        node.right = self.__construct_tree_helper(
            best_feature_split.right_split_indexes,
            depth + 1
        )

        if node.is_leaf():
            node.answer = self.__get_answer(
                observation_indexes
            )

        return node

    def __get_answer(self, observation_indexes):
        labels = self.y[observation_indexes, :]

        if self.__task == 'classification':
            ones = labels.sum()
            threshold = labels.shape[0] / 2

            if ones > threshold:
                return 1
            elif ones < threshold:
                return 0
            else:
                return self.random_state.randint(0, 2)
        else:
            return np.mean(labels)

    def __construct_tree(self):
        observation_indexes = np.arange(self.y.shape[0])

        self.root = self.__construct_tree_helper(
            observation_indexes,
            1
        )

        if self.root is None:
            self.root = DecisionTree.Node(
                None,
                None,
                observation_indexes
            )

            self.answer = self.__get_answer(
                observation_indexes
            )