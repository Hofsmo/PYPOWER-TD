"""Module for testing algorithms."""
from collections import namedtuple
import pytest
import sympy
import control
import numpy as np
from pypiw import systems, algorithms


@pytest.fixture(scope='session')
def tf():
    """Create transfer function."""
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


@pytest.fixture(scope='session')
def data_vec():
    """Create data for testing"""
    data = namedtuple('data', 't x c_tf y')
    data.t = np.arange(0, 10, 0.02)
    data.x = np.ones(len(data.t))
    data.parameters = {'T1': 2.0, 'T2': -3.0}
    data.c_tf = control.tf([2.0, 1.0], [-3.0, 1.0])
    _, data.y, _ = control.forced_response(data.c_tf, data.t, data.x)

    return data


@pytest.fixture(scope='session')
def ga(data_vec, tf):
    """ Create the algorithm object to test"""
    return algorithms.Ga(data_vec.x, data_vec.y, data_vec.t, tf, -5, 5)


def test_compare(ga, tf, data_vec):
    """Test the compare method"""
    np.testing.assert_almost_equal(
        super(algorithms.Ga, ga).compare(data_vec.parameters), 0)


def test_ga(ga, tf):
    ga.identify()
    np.testing.assert_almost_equal(ga.identified_parameters()['T2'], -3.0, 0)
