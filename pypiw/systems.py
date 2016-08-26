from abc import ABCMeta, abstractmethod
import six
import sympy
import control
import numpy as np


@six.add_metaclass(ABCMeta)
class SystemBase():
    """
    Base class for system representations
    """

    @abstractmethod
    def time_response(self):
        """
        This method calculates the time response of the system
        """
        pass


class Tf(SystemBase):
    """
    Class for transfer function representation
    """
    def __init__(self, sys):
        self.sys = sys
        s = sympy.symbols('s')
        try:
            num = sympy.degree(sympy.numer(sys), s)
        except TypeError:
            print("System is not a sympy object")

        den = sympy.degree(sympy.denom(sys), s)

        if den < num:
            raise ValueError("System is not proper")

        self.f = sympy.lambdify(
            self.sys.atoms(
                sympy.Symbol).difference({s}), self.sys, "numpy")

    def num_den(self, parameters):
        # Extract the numerator and denominator
        temp = self.f(*parameters)

        # The control toolbox does not understand sympy integers. Therefore
        # it is necessary to convert
        num = np.asarray(sympy.Poly(sympy.numer(temp)).all_coeffs(), float)
        den = np.asarray(sympy.Poly(sympy.denom(temp)).all_coeffs(), float)

        return num, den

    def time_response(self, parameters, x, t):
        """
        Method that calculates the time response of the system
        Input:
            parameters: Value of the parameters in the system
            x: Input vector
            t: Time vector
        Output:
            numpy array containint the time response
        """
        num, den = self.num_den(parameters)

        if len(num) < len(den):
            parameters[-1] = 0.0001
            num, den = self.num_den(parameters)

        # _, y, _ = control.forced_response(
        #    control.tf(num, den), t,  x)

        _, y, _ = control.forced_response(
            control.tf([parameters[0], 1], [parameters[1], 1]), t,  x)
        return y