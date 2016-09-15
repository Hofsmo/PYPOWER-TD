import pytest
import sympy
import control
import numpy as np
from collections import namedtuple
from pypiw import systems, algorithms


@pytest.fixture(scope='session')
def tf():
    s, T1, T2 = sympy.symbols('s T1 T2')

    return systems.Tf((1+s*T1)/(1+s*T2))


@pytest.fixture(scope='session')
def data_vec():
    data = namedtuple('data', 't x c_tf y')
    data.t = np.arange(0, 10, 0.02)
    data.x = np.ones(len(data.t))
    data.c_tf = control.tf([2.0, 1.0], [-3.0, 1.0])
    _, data.y, _ = control.forced_response(data.c_tf, data.t, data.x)

    return data


def test_ga(tf, data_vec):
    ga = algorithms.Ga(data_vec.x, data_vec.y, data_vec.t, tf, -5, 5)
    ga.identify()

    if tf.atoms_list[0] == 'T1':
        np.testing.assert_almost_equal(ga.hof[0][0], 2.0, 1)
    else:
        np.testing.assert_almost_equal(ga.hof[0][0], -3.0, 1)
