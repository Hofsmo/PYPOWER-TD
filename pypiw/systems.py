"""Module containing the different system representations."""

from abc import ABCMeta, abstractmethod
import six
import sympy
import control
import numpy as np


@six.add_metaclass(ABCMeta)
class SystemBase():
    """Base class for system representations."""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def time_response(self, parameters, x, t):
        """This method calculates the time response of the system."""
        pass


class Tf(SystemBase):
    """Class for transfer function representation."""
    def __init__(self, sys):
        super(Tf, self).__init__()
        self.sys = sys
        s = sympy.symbols('s')
        num = sympy.degree(sympy.numer(sys), s)

        den = sympy.degree(sympy.denom(sys), s)

        if den < num:
            raise ValueError("System is not proper")

        self.atoms = self.sys.atoms(sympy.Symbol).difference({s})
        self.atoms_list = [str(atom) for atom in self.atoms]
        self.n_atoms = len(self.atoms)
        self.f = sympy.lambdify(self.atoms, self.sys, "numpy")

    @property
    def sys(self):
        """System property"""
        return self._sys

    @sys.setter
    def sys(self, value):
        """Setter for sys"""
        if not isinstance(value, tuple(sympy.core.all_classes)):
            raise TypeError("sys is not a sympy object")
        self._sys = value

    def num_den(self, parameters):
        """Returns the numerator and denominator coefficients.

        Method that takes in the estimated parameters and calculates
        the denominator and numerator coefficients

        Args:
           parameters: estimated parameters

        Returns:
            num: numerator coefficients
            den: denominator coefficients
        """
        if isinstance(parameters, dict):
            par = [None]*len(self.atoms_list)
            for idx, atom in enumerate(self.atoms_list):
                par[idx] = parameters[atom]
        else:
            par = parameters

        # Extract the numerator and denominator
        temp = self.f(*par)

        # The control toolbox does not understand sympy integers. Therefore
        # it is necessary to convert
        num = np.asarray(sympy.Poly(sympy.numer(temp)).all_coeffs(), float)
        den = np.asarray(sympy.Poly(sympy.denom(temp)).all_coeffs(), float)

        return num, den

    def time_response(self, parameters, x, t):
        """Method that calculates the time response of the system.

        Args:
            parameters: Value of the parameters in the system
            x: Input vector
            t: Time vector

        Returns:
            numpy array containint the time response
        """
        num, den = self.num_den(parameters)

        if len(num) < len(den):
            parameters[-1] = 0.0001
            num, den = self.num_den(parameters)

        _, y, _ = control.forced_response(
            control.tf(num, den), t, x)

        return y
