cimport numpy as np
import numpy as np

ctypedef size_t Index
ctypedef size_t TargetData
ctypedef double FeaturesData

# numpy buffer types
ctypedef np.uint64_t NPIndex
ctypedef np.uint64_t NPTargetData
ctypedef np.float64_t NPFeaturesData

# array buffer types
cdef str AIndex = 'Q'
cdef str ATargetData = 'Q'
cdef str AFeaturesData = 'd'