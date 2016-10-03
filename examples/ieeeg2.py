import os
import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypiw import systems, algorithms


def main():
    here = os.path.dirname(__file__)
    fname = os.path.join(here, 'Data/in.csv')
    df = pd.read_csv(fname)
    s, K, T1, T2, T3, T4 = sympy.symbols('s K T1 T2 T3 T4')

    #tf = -K*(1+s*T2)*(1-s*T4)/((1+s*T1)*(1+s*T3)*(1+0.5*s*T4))
    tf = -K*(1+s*T2)/((1+s*T1)*(1+s*T3))

    sys = systems.Tf(tf)
    # answer = {'K': 1.0, 'T1': 2.0, 'T2': 3.0, 'T3': 4.0, 'T4': 5.0}
    # y = sys.time_response(answer, x, t)

    # ga = algorithms.Ga(x, y, t, sys, 0, 6.0)
    ga = algorithms.Ga(df['in'], df['out'], df['time'], sys, 0, 150, nind=1000, sigma=5)
    ga.identify(verbose=True)
    print(ga.identified_parameters())
    plt.plot(df['time'], df['out'], 'blue', df['time'], sys.time_response(ga.identified_parameters(), df['in'], df['time']), 'red')
    plt.show()


if __name__ == "__main__":
    main()
