from abc import ABCMeta, abstractmethod
import six
from deap import base, creator, tools
import random
import numpy as np


@six.add_metaclass(ABCMeta)
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

    @abstractmethod
    def identify(self, conf=[]):
        """
        Function to perform the identification
        Input:
            conf: Configuration structure for the identification
        """
        pass


class PyPiWGa(PyPiW):
    """
    Class that implements parameter identification for transfer functions
    """
    def __init__(self, in_data=[], out_data=[], ts=[], sys=[], conf=[]):
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
        self.sys = sys

    def compare(self, individual):
        return np.std(
            self.sys.time_response(
                individual, self.in_data, self.ts) - self.out_data)

    def identify(self, conf=[], v=False):
        # If no configuration is given use the one already
        if not conf:
            if not self.conf:
                raise ValueError("Configuration is empty")
            else:
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

        pop = toolbox.population(n=conf.NGEN)

        if v:
            print("Start of evolution")

        fitnesses = [self.compare(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("Evaluated %i individuals" % len(pop))

        # Begin the evolution
        for g in range(conf.NGEN):
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # Cross two individuals with probability CXPB
                if random.random() < conf.CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children must be calculated later
                    del child1.fitness.values
                    del child2.fitness.values

                for mutant in offspring:

                    # mutate an individual with probability MUTPB
                    if random.random() < conf.MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.compare(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print(" Min %s" % min(fits))
            print(" Max %s" % max(fits))
            print(" Avg %s" % mean)
            print(" Std %s" % std)

        print("-- End of (successful) evaluation --")

        best_ind = tools.selBest(pop, 1)[0]
        print(best_ind)
