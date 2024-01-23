import sys, numpy
sys.modules["scipy.random"] = numpy.random
import pybrain3
sys.modules[__name__] = pybrain3