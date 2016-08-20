from deap import base, creator, tools
import control
import random
import numpy as np

# Make it a minimization problem
creator.create("FitnessMax", base.Fitness, weights=(-1.0))
creator.create("Individual", fitness=creator.FitnessMax)

# Create functions for creating individuals and population
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, 0.0, 10.0)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 2)
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)

# Define root mean square error
def rmse(y, y0):
    return np.sqrt(((y - y0) ** 2).mean())

# Define the fitness function
def compare(individual, y):
    sys = control.tf([individual[0], 1], [individual[1]], 1])
    return rmse(np.array(control.step_response(sys)),np.array(y))

toolbox.register("evaluate", compare, individual, y)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, turnsize=3)

def main():
    random.seed(64)

# Create the data to run the optimization on
_, y = control.step_response(tf([-2.0, 1.0], [3.0, 1.0]))

    pop = toolbox.population(n=5)
    hof = tools.HallOfFame(1)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("Start of evolution")

    fitnesses = [compare(ind, y) for ind in individuals]



