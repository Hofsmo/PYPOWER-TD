"""
Module for testing the systems module
"""
from collections import namedtuple
import pytest
import sympy
import control
from pypiw import systems
import numpy as np


@pytest.fixture(scope='session')
def tf():
    """
    Create the transfer funcion to use for the tests
    """
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


@pytest.fixture(scope='session')
def data_vec():
    """
    Create the data to use in the tests
    """
    temp = namedtuple('temp', 't x c_tf y')
    temp.t = np.arange(0, 0.5, 0.2)
    temp.x = np.ones(len(temp.t))
    temp.num = [2.0, 1.0]
    temp.den = [-3.0, 1.0]
    temp.c_tf = control.tf(temp.num, temp.den)
    _, temp.y, _ = control.forced_response(temp.c_tf, temp.t, temp.x)

    return temp


def test_init(tf):
    assert(isinstance(tf.sys, tuple(sympy.core.all_classes)))


def test_num_den(tf):
    num, den = tf.num_den([2.0, -3.0])

    if tf.atoms_list[0] == 'T1':
        np.testing.assert_almost_equal(num[0], 2.0)
    else:
        np.testing.assert_almost_equal(num[0], -3.0)


def test_time_response(tf, data_vec):

    # The order of the parameters is different in some versions of Python.
    # This does not matter for the identification
    if tf.atoms_list[0] == 'T2':
        y_sys = tf.time_response([-3.0, 2.0], data_vec.x, data_vec.t)
    else:
        y_sys = tf.time_response([2.0, -3.0], data_vec.x, data_vec.t)

    np.testing.assert_allclose(data_vec.y, y_sys)
