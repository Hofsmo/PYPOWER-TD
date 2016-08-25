import pytest
import sympy
from pypiw import systems
import numpy as np
import control


@pytest.fixture(scope='session')
def sys():
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


def test_init(sys):
    assert(isinstance(sys.sys, tuple(sympy.core.all_classes)))


def test_time_response(sys):
    t = np.arange(0, 0.5, 0.2)
    x = np.ones(len(t))
    tf = control.tf([2, 1], [-3, 1])
    _, y, _ = control.forced_response(tf, t, x)

    y_sys = sys.time_response([-3, 2], x, t)

    np.testing.assert_array_equal(y, y_sys)
