import sys
import pytest
import sympy
import control
from pypiw import systems
import numpy as np


@pytest.fixture(scope='session')
def tf():
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


def test_init(tf):
    assert(isinstance(tf.sys, tuple(sympy.core.all_classes)))


def test_time_response(tf):
    t = np.arange(0, 0.5, 0.2)
    x = np.ones(len(t))
    c_tf = control.tf([2.0, 1.0], [-3.0, 1.0])
    _, y, _ = control.forced_response(c_tf, t, x)

    # The order of the parameters is different in some versions of Python.
    # This does not matter for the identification
    if sys.version_info >= (3, 0):
        y_sys = tf.time_response([-3.0, 2.0], x, t)
    else:
        y_sys = tf.time_response([2.0, -3.0], x, t)

    np.testing.assert_allclose(y, y_sys)
