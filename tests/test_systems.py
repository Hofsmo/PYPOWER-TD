import sys
import pytest
import sympy
import control
from collections import namedtuple
from pypiw import systems, algorithms
import numpy as np


@pytest.fixture(scope='session')
def tf():
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


@pytest.fixture(scope='session')
def data_vec():
    data = namedtuple('data', 't x c_tf y')
    data.t = np.arange(0, 0.5, 0.2)
    data.x = np.ones(len(data.t))
    data.c_tf = control.tf([2.0, 1.0], [-3.0, 1.0])
    _, data.y, _ = control.forced_response(data.c_tf, data.t, data.x)

    return data


def test_init(tf):
    assert(isinstance(tf.sys, tuple(sympy.core.all_classes)))


def test_time_response(tf, data_vec):

    # The order of the parameters is different in some versions of Python.
    # This does not matter for the identification
    if sys.version_info >= (3, 0):
        y_sys = tf.time_response([-3.0, 2.0], data_vec.x, data_vec.t)
    else:
        y_sys = tf.time_response([2.0, -3.0], data_vec.x, data_vec.t)

    np.testing.assert_allclose(data_vec.y, y_sys)


def test_ga(tf, data_vec):
    ga = algorithms.Ga(data_vec.x, data_vec.y, data_vec.t, tf, -5, 5)
    ga.identify()
    np.testing.assert_allclose([-3.0, 2.0], ga.best_ind)
