from pypiw import pypiw


def test_pypiw():
    a = pypiw.PyPiW([],[],[],[])
    assert(not a.in_data)
    assert(not a.out_data)
