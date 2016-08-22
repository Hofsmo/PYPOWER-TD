from deap import base, creator, tools, algorithms
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


# Create the data to run the optimization on
t, y = control.step_response(control.tf([-2.0, 1.0], [3.0, 1.0]))


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

toolbox.register("evaluate", compare, y=y, t=t)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.8, low=-10, up=10)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)

    pop = toolbox.population(n=300)

    hof = tools.HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.1, mutpb=0.1, ngen=10,
                                   stats=stats, halloffame=hof)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    sys = control.tf([best_ind[0], 1], [best_ind[1], 1])

    t, y_est = control.step_response(sys)

    plt.plot(t, y, 'r',  y_est, 'b')
    plt.show()

if __name__ == "__main__":
    main()
