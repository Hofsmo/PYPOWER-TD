"""Module containing the different system representations."""

from abc import ABCMeta, abstractmethod
import six
import sympy
import control
import os.path
import numpy as np
import fmipp
import shutil
import urlparse



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

    def step_response(self, parameters, t=None):
        """Method that calculates the step response of a system.

        Args:
            parameters: Values of the paramateres describing the system.
            This can either be a list or a dict. If it is a dict the keys
            should correspond to the names of the parameters.
            t: The time vector to calulate the response for

        Returns:
            y: The step response.
            t: the time vector.
            """
        num, den = self._num_den(parameters)
        t, y, _ = control.step_response(control.tf(num, den), t)

        return t, y


class ModelicaSystem(SystemBase):
    '''
    Class implementing the Modelica FMUs as systems for the identification.
    '''

    def __init__(self, file_path, model_name, logging_on=False,event_search_precision=1e-7,
                 integrator_type=fmipp.rk):
        '''
        Instantiates a ModelicaSystem object and loads/compiles an FMU.
        :param file_path: File path to the modelica/fmu file.
        :param model_name: Name of the model in the FMU.
        '''
        self.file_path = file_path
        self.model_name = model_name

        assert os.path.exists(file_path), 'File path does not exist!'

        # Extract the FMU
        self.extracted_fmu = fmipp.extractFMU(file_path, os.path.dirname(file_path))
        self.logging_on = logging_on
        self.event_search_precision = event_search_precision
        self.integrator_type = integrator_type
        self.fmu = fmipp.FMUModelExchangeV2(self.extracted_fmu, self.model_name, self.logging_on,
                                        False,self.event_search_precision, self.integrator_type)

    def time_response(self, parameters, x, t):
        '''
        Method that returns the response of the FMU
        :param parameters: Model's parameters.
        :param x: Input to the model.
        :param t: Time vector
        :return y: Returns the output of the model.
        '''

        status = self.fmu.instantiate(self.model_name)  # instantiate model
        assert status == fmipp.fmiOK, "Could not instantiate the model"  # check status

        self.setParams(parameters)  # set model parameters

        status = self.fmu.initialize()  # initialize model
        assert status == fmipp.fmiOK, "Could not initialize the model"  # check status

        y = np.empty(len(t))

        for index, t_step in enumerate(t):
            self.fmu.setRealValue("u", x[index])
            self.fmu.integrate(t_step)  # integrate model
            y[index] = self.fmu.getRealValue("y")  # retrieve output variable 'x'

        return y

    def setParams(self, params):
        '''
        A function that sets the parameters in the fmu form the dictionary params.
        Currently, only real parameters are supported.
        :param params: Dictionary of parameters to be set in the fmu
        '''

        for name, value in params.items():
            self.fmu.setRealValue(name, value)


    def cleanFMU(self):
        '''
        Removes the directory containing extracted FMU.
        To be implemented.
        :return:
        '''
        p = urlparse.urlparse(self.extracted_fmu)
        path = p.path
        print path

