"""
Main package for pdaggerq

_pdagerq is the name of the pybind module.

If it was also named pdaggerq we would have a namespace collision.

When importing pdaggerq it must go grab all the modules from _pdaggerq
"""
from ._pdaggerq import *
