from deap import base, creator, tools, algorithms
import random
import numpy as np


class Ga():
    """
    Class that implements the genetic algorithm presented in DEAP one max
    problem.
    """
    def __init__(self, in_data=[], out_data=[], t=[], sys=[], lower=[],
                 upper=[], ngen=40, nind=300, cxpb=0.5, mutpb=0.1, indpb=0.5,
                 tournsize=3):
        """
        Initialize the object
        Input:
            in_data: Array of in data
            out_data: Array of response data
            t: Time vector
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
        self.t = t
        self.sys = sys
        self.lower = lower
        self.upper = upper
        self.ngen = ngen
        self.nind = nind
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.tournsize = tournsize
        self.best_ind = []

        # Make it a minimization problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list,  fitness=creator.FitnessMin)

        # Create functions for creating individuals and population
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool",
                              random.uniform, self.lower, self.upper)
        self.toolbox.register("individual",
                              tools.initRepeat,
                              creator.Individual, self.toolbox.attr_bool, 2)
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
                individual, self.in_data, self.t) - self.out_data),

    def identify(self, v=False):

        # if v:
            # print("Start of evolution")

        # fitnesses = [self.compare(ind) for ind in self.pop]
        # for ind, fit in zip(self.pop, fitnesses):
            # ind.fitness.values = fit

        # print("Evaluated %i individuals" % len(self.pop))

        # # Begin the evolution
        # for g in range(self.ngen):
            # print("-- Generation %i --" % g)

            # # Select the next generation individuals
            # offspring = self.toolbox.select(self.pop, len(self.pop))
            # # Clone the selected individuals
            # offspring = list(map(self.toolbox.clone, offspring))

            # for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # # Cross two individuals with probability CXPB
                # if random.random() < self.cxpb:
                    # self.toolbox.mate(child1, child2)

                    # # fitness values of the children must be calculated later
                    # del child1.fitness.values
                    # del child2.fitness.values

                # for mutant in offspring:

                    # # mutate an individual with probability MUTPB
                    # if random.random() < self.mutpb:
                        # self.toolbox.mutate(mutant)
                        # del mutant.fitness.values

            # # Evaluate the individuals with an invalid fitness
            # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # fitnesses = [self.compare(ind) for ind in invalid_ind]
            # for ind, fit in zip(invalid_ind, fitnesses):
                # ind.fitness.values = fit

            # print("Evaluated %i individuals" % len(invalid_ind))

            # # The population is entirely replaced by the offspring
            # self.pop[:] = offspring

            # # Gather all the fitnesses in one list and print the stats
            # fits = [ind.fitness.values[0] for ind in self.pop]

            # length = len(self.pop)
            # mean = sum(fits) / length
            # sum2 = sum(x*x for x in fits)
            # std = abs(sum2 / length - mean**2)**0.5

            # if v:
                # print(" Min %s" % min(fits))
                # print(" Max %s" % max(fits))
                # print(" Avg %s" % mean)
                # print(" Std %s" % std)

        # if v:
            # print("-- End of (successful) evaluation --")
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("std", np.std)
        self.pop = algorithms.eaSimple(self.pop, self.toolbox, cxpb=self.cxpb,
                                       mutpb=self.mutpb, ngen=self.ngen,
                                       stats=stats, verbose=True)
        self.best_ind = tools.selBest(self.pop, 1)[0]
