import sympy
from pypiw import systems


def main():
    s, K, T1, T2, T3, T4 = sympy.symbols('s K T1 T2 T3 T4')

    tf = -K*(1+s*T2)*(1-s*T4)/((1+s*T1)*(1+s*T3)*(1+0.5*s*T4))

    sys = systems.Tf(tf)
    print(sys.num_den({'K': 1.0, 'T1': 2.0, 'T2': 3.0, 'T3': 4.0, 'T4': 5.0}))


if __name__ == "__main__":
    main()
