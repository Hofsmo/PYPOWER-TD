"""
Module for testing the systems module
"""
from collections import namedtuple
import pytest
import sympy
import control
from pypiw import systems
import numpy as np


@pytest.fixture(scope='function')
def tf():
    """Create the transfer funcion to use for the tests."""
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


@pytest.fixture(scope='session')
def data_vec():
    """Create the data to use in the tests"""
    temp = namedtuple('temp', 't x c_tf y')
    temp.t = np.arange(0, 0.5, 0.2)
    temp.x = np.ones(len(temp.t))
    temp.num = [2.0, 1.0]
    temp.den = [-3.0, 1.0]
    temp.parameters = {'T1': 2.0, 'T2': -3.0}
    temp.c_tf = control.tf(temp.num, temp.den)
    _, temp.y, _ = control.forced_response(temp.c_tf, temp.t, temp.x)

    return temp


def test_sys_get(tf):
    """Test if sys returns correct object"""
    assert isinstance(tf.sys, tuple(sympy.core.all_classes))


def test_sys_error(tf):
    """Test if exception is raised"""
    with pytest.raises(Exception):
        tf.sys = True


def test_not_proper():
    """Test what happens if the system is not proper."""
    s = sympy.symbols('s')
    with pytest.raises(Exception):
        systems.Tf(s)


def test_num_den(tf, data_vec):
    """Check if num and den are correctly returned."""
    num, den = tf.num_den(data_vec.parameters)

    assert num[0] == data_vec.num[0] and den[0] == data_vec.den[0]


def test_time_response(tf, data_vec):
    """Check if time response is calculated correctly."""

    y_sys = tf.time_response(data_vec.parameters, data_vec.x, data_vec.t)

    np.testing.assert_allclose(data_vec.y, y_sys)
