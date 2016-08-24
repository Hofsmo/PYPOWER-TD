from deap import base, creator, tools
import random


class PyPiW():
    """
    Class that holds the data and configuration for projects
    """
    def __init__(self, in_data=[], out_data=[], ts=[], conf=[]):
        """
        Constructor of PyPiW
        Input:
            in_data: Array of in data
            out_data: Array of response data
            ts: Time step
            conf: Configuarion structure for the identification
        """
        self.in_data = in_data
        self.out_data = out_data
        self.ts = ts
        self.conf = conf

    def identify(self, conf=[]):
        """
        Function to perform the identification
        Input:
            conf: Configuration structure for the identification
        """
        return []


class PyPiWTf(PyPiW):
    """
    Class that implements parameter identification for transfer functions
    """
    def __init__(self, in_data=[], out_data=[], ts=[], tf=[], conf=[]):
        """
        Constructor of PyPiWTf
        Input:
            in_data: Array of in data
            out_data: Array of response data
            ts: Time step
            tf: Transfer function given as sympy
            conf: Configuarion structure for the ga
        """
        PyPiW.__init__(self, in_data, out_data, ts, conf)
        self.tf = tf

    def identify(self, conf=[]):
        if not conf:
            conf = self.conf

        # Make it a minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,  fitness=creator.FitnessMin)

        # Create functions for creating individuals and population
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.uniform, conf.lower, conf.upper)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, 2)
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)
