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
        # Define the laplace operator
        s = sympy.symbols('s')

        # Create lambda to generate transfer function
        try:
            f = sympy.lambdify(
                self.sys.atoms(
                    sympy.Symbol).difference({s}), self.sys, "numpy")
        except TypeError:
            print("Could not read system")

        # Extract the numerator and denominator
        temp = f(*parameters)

        # The control toolbox does not understand sympy integers. Therefore
        # it is necessary to convert
        num = np.asarray(sympy.Poly(sympy.numer(temp)).all_coeffs(), float)
        den = np.asarray(sympy.Poly(sympy.denom(temp)).all_coeffs(), float)

        _, y, _ = control.forced_response(
            control.tf(num, den), t,  x)

        return y
