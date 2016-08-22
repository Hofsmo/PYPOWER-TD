from deap import base, creator, tools
import control
import random
import numpy as np
import matplotlib.pyplot as plt

# Make it a minimization problem
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list,  fitness=creator.FitnessMax)


def create_ind():
    i = random.uniform(-10.0, 10.0)

    if i >= -0.0001 and i <= 0.0001:
        i = 0.001

    return i

# Create functions for creating individuals and population
toolbox = base.Toolbox()
toolbox.register("attr_bool", create_ind)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 2)
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)


# Define the fitness function
def compare(individual, y, t):
    num = individual[0]
    den = individual[1]

    if den >= -0.00001 and den <= 0.00001:
        den = 0.01
        individual[1] = den

    sys = control.tf([num, 1], [den, 1])
    _, yfit = control.step_response(sys, t)

    return np.std(yfit - y),

toolbox.register("evaluate", compare)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)

    # Create the data to run the optimization on
    t = np.arange(0, 20, 0.02)
    _, y = control.step_response(control.tf([-2.0, 1.0], [3.0, 1.0]), t)

    pop = toolbox.population(n=100)

    CXPB, MUTPB, NGEN = 0.5, 0.05, 100

    print("Start of evolution")

    fitnesses = [compare(ind, y, t) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # Cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children must be calculated later
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [compare(ind, y, t) for ind in invalid_ind]
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

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    sys = control.tf([best_ind[0], 1], [best_ind[1], 1])

    t, y_est = control.step_response(sys, t)

    plt.plot(t, y, 'r',  y_est, 'b')
    plt.show()

if __name__ == "__main__":
    main()
