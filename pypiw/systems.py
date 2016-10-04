"""Module containing the different system representations."""

from abc import ABCMeta, abstractmethod
import six
import sympy
import control
import os.path
import numpy as np
import fmipp



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
        # Extract the numerator and denominator
        temp = self.f(*parameters)

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

class ModelicaSystem(SystemBase):
    '''
    Class implementing the Modelica FMUs as systems for the identification.
    '''
    def __init__(self, file_path, model_name, logging_on=False, stop_before_event=False, event_search_precision=1e-5,
                 integrator_type=fmipp.rk):
        '''
        Instantiates a ModelicaSystem object and loads/compiles an FMU.
        :param file_path: File path to the modelica/fmu file.
        :param model_name: Name of the model in the FMU.
        '''
        self.file_path = file_path
        self.model_name = model_name
        self.compiled = compiled

        assert os.path.exists(file_path), 'File path does not exist!'

        # Extract the FMU
        self.extracted_fmu = fmipp.extractFMU(file_path, os.path.dirname(file_path))
        self.logging_on = logging_on
        self.stop_before_event = stop_before_event
        self.event_search_precision = event_search_precision
        self.integrator_type = integrator_type
        self.fmu = fmipp.FMUModelExchangeV1(self.uri_to_extracted_fmu, self.model_name, self.logging_on,
                                            self.stop_before_event, self.event_search_precision, self.integrator_type)

        status = self.fmu.instantiate("my_test_model_1")  # instantiate model
        assert status == fmipp.fmiOK, 'The FMU could not be instantiated'

    def time_response(self, parameters, x, t):
        '''
        Method that returns the response of the FMU
        :param parameters: Model's parameters.
        :param x: Input to the model.
        :param t: Time vector
        :return y: Returns the output of the model.
        '''

        result = self.fmu.simulate(start_time = 0.0, final_time = t[-1])

        return result['y']
