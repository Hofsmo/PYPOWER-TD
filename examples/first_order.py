from pypiw import algorithms, systems
import numpy as np
import control
import sympy
# Define the fitness function


def main():
    t = np.arange(0, 20, 0.02)
    x = np.ones(len(t))
    tf = control.tf([2.0, 1.0], [-3.0, 1.0])
    _, y, _ = control.forced_response(tf, t, x)

    s, T1, T2 = sympy.symbols('s T1 T2')
    sys = (1+s*T1)/(1+s*T2)

    tf = systems.Tf(sys)
    ga = algorithms.Ga(x, y, t, tf, -5.0, 5.0)
    ga.identify(v=True)
    print(ga.best_ind)

if __name__ == "__main__":
    main()
