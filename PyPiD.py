from deap import import base, creator, tools
import random

creator.create("FitnessMax", base.Fitness, weights=(1.0))
creator.create("Individual", fitness=creator.FitnessMax)

