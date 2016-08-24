import sys
import os

# PEP8 requires all imports to be at the top.
sys.path.append(os.path.abspath(".."))
try:
    from pypiw import pypiw
except:
    raise


def test_pypiw():
    a = pypiw.PyPiW()
    assert(not a.in_data)
    assert(not a.out_data)
