"""
Module containing the available algorithms
"""
from abc import ABCMeta, abstractmethod
import random
import six
from deap import base, creator, tools, algorithms
import numpy as np


@six.add_metaclass(ABCMeta)
class AlgorithmBase():
    """
    Base class for algorithms
    """
    def __init__(self, in_data, out_data, time, sys, lower, upper):
        """
        Constructor for the algorithm base class
        """
        self.in_data = in_data
        self.out_data = out_data
        self.time = time
        self.sys = sys
        self.lower = lower
        self.upper = upper

    @abstractmethod
    def identify(self, verbose):
        """All algortihms should provide an identify method."""
        pass

    @abstractmethod
    def identified_parameters(self):
        """ All algorithms should be able to return the best individual."""
        pass

    def compare(self, parameter):
        """Method that compares the individuals with the correct value."""
        return np.std(
            self.sys.time_response(
                parameter, self.in_data, self.time) - self.out_data),

    def check_bounds(self, min, max):
        """Check min max for parameter."""
        def decorator(func):
            """Decorator function."""
            def wrapper(*args, **kargs):
                """Wrapper function."""
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max:
                            child[i] = max-0.00001
                        elif child[i] < min:
                            child[i] = min+0.00001
                return offspring
            return wrapper
        return decorator


class Ga(AlgorithmBase):
    """
    Class that implements the genetic algorithm presented in DEAP one max
    problem.
    """
    def __init__(self, in_data, out_data, time, sys, lower, upper,
                 ngen=40, nind=300, cxpb=0.5, alpha=0.5, mutpb=0.2, mu=0,
                 sigma=0.1, indpb=0.5, tournsize=3):
        """
        Initialize the object
        Input:
            in_data: Array of in data
            out_data: Array of response data
            time: Time vector
            sys: Transfer function given as sympy
            lower: Lower bound of the parameters
            upper: Upper bound of the parameters
            ngen: Number of generations default=40
            nind: Number of individual default=100
            cxpb: Probability of mutation deafault=0.5
            mutp: Probability of mutation default=0.1
            indpb: Probability of mutation of each gene deafult=0.5
            mating: The mating strategy default is blended crossover
            mutation: The mutation strategy the default is gaussian

        """
        super(Ga, self).__init__(in_data, out_data, time, sys, lower, upper)
        self.ngen = ngen
        self.nind = nind
        self.cxpb = cxpb
        self.alpha = alpha
        self.mutpb = mutpb
        self.mu = mu
        self.sigma = sigma
        self.indpb = indpb
        self.tournsize = tournsize
        self.hof = tools.HallOfFame(1)

        # Make it a minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create functions for creating individuals and population
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool",
                              random.uniform, self.lower, self.upper)
        self.toolbox.register("individual",
                              tools.initRepeat,
                              creator.Individual,
                              self.toolbox.attr_bool,
                              self.sys.n_atoms)
        self.toolbox.register("population",
                              tools.initRepeat, list, self.toolbox.individual)

        # Register the crossover and mutation functions
        self.toolbox.register("mate", tools.cxBlend, alpha=self.alpha)
        self.toolbox.register("mutate", tools.mutGaussian, mu=self.mu,
                              sigma=self.sigma, indpb=self.indpb)

        self.toolbox.decorate("mate",
                              self.check_bounds(self.lower, self.upper))
        self.toolbox.decorate("mutate",
                              self.check_bounds(self.lower, self.upper))

        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.tournsize)

        self.pop = self.toolbox.population(n=self.nind)
        self.toolbox.register("evaluate", self.compare)

    def identify(self, algorithm='simple', verbose=False):
        """
        Function performing the idenfication
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("std", np.std)
        stats.register("min", np.min)

        if algorithm == 'simple':
            self.pop = algorithms.eaSimple(self.pop, self.toolbox,
                                           cxpb=self.cxpb,
                                           mutpb=self.mutpb, ngen=self.ngen,
                                           stats=stats, verbose=verbose,
                                           halloffame=self.hof)
        elif algorithm == 'generate':
            self.pop = algorithms.eaGenerateUpdate(self.toolbox, self.ngen,
                                                   self.hof)
        else:
            raise ValueError("No such algorithm")

    def identified_parameters(self):
        """Return the best identified parameter."""
        return {key: value for (key, value) in zip(self.sys.atoms_list,
                                                   self.hof[0])}
