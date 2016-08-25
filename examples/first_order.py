import control
import random
import numpy as np
import matplotlib.pyplot as plt


# Define the fitness function


def main():
    random.seed(64)

    # Create the data to run the optimization on
    t = np.arange(0, 20, 0.02)
    _, y = control.step_response(control.tf([-2.0, 1.0], [3.0, 1.0]), t)


    #CXPB, MUTPB, NGEN = 0.5, 0.5, 40


    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


    t, y_est = control.step_response(sys, t)

    plt.plot(t, y, 'r', t,  y_est, 'b')
    plt.show()

if __name__ == "__main__":
    main()
