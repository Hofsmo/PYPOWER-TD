
from pypiw import systems


def test_init():
    a = systems.Tf(sys=1)
    assert(a.sys == 1)
