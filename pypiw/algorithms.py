"""
Module containing the available algorithms
"""
import random
from deap import base, creator, tools, algorithms
import numpy as np


class Ga(object):
    """
    Class that implements the genetic algorithm presented in DEAP one max
    problem.
    """
    def __init__(self, in_data, out_data, time, sys, lower, upper,
                 ngen=40, nind=300, cxpb=0.5, mutpb=0.2, indpb=0.5,
                 tournsize=3):
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
        self.in_data = in_data
        self.out_data = out_data
        self.time = time
        self.sys = sys
        self.lower = lower
        self.upper = upper
        self.ngen = ngen
        self.nind = nind
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.tournsize = tournsize
        self.hof = tools.HallOfFame(1)

        # Make it a minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,  fitness=creator.FitnessMin)

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
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1,
                              indpb=self.indpb)

        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.tournsize)

        self.pop = self.toolbox.population(n=self.nind)
        self.toolbox.register("evaluate", self.compare)

    def compare(self, individual):
        """ function that compares the individuals with the correct value
        """
        return np.std(
            self.sys.time_response(
                individual, self.in_data, self.time) - self.out_data),

    def identify(self, verbose=False):
        """
        Function performing the idenfication
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("std", np.std)
        stats.register("min", np.min)

        self.pop = algorithms.eaSimple(self.pop, self.toolbox, cxpb=self.cxpb,
                                       mutpb=self.mutpb, ngen=self.ngen,
                                       stats=stats, verbose=verbose,
                                       halloffame=self.hof)
