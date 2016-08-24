import sys
import os

# PEP8 requires all imports to be at the top.
sys.path.append(os.path.abspath('..'))
try:
    from pypiw import pypiw
except:
    raise


def test_init():
    a = pypiw.PyPiW()
    assert(a.data != 1)
