from pypiw import systems
import numpy as np
import os
import inspect


def main():
    '''
    Example demonstrating how to simulate the Modelica file
    '''

    module_path = os.path.dirname(inspect.getfile(inspect.currentframe()))  # script directory
    #  Creating the system object
    fmu_sys = systems.ModelicaSystem(os.path.join(module_path, 'FMUs/model3.fmu'), 'model3')

    # Creating the time and input vectors
    t = np.arange(0, 4, 0.5)
    x = np.ones(len(t))

    # Create the dictionary of parameters
    params = {'gain1.k': 2}
    y = fmu_sys.time_response(params, x, t)

    # Printing time, input and output

    for idx, t_step in enumerate(t):
        print "Time:", t_step, "- Input:", x[idx], "- Output:", y[idx]

    fmu_sys.cleanFMU()

if __name__ == "__main__":
    main()
