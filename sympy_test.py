from sympy import symbols, Poly

# Define s as a symbol
s = symbols('s')

# Define input w as symbol
w = symbols('w')

# Define helper points
c = symbols('c')

Tf, Tr, r, Tg, R, D, Tw, At = symbols(
    'Tf Tr r Tg R D Tw At')

block1 = 1/(1+Tf*s)
block2 = (1+Tr*s)/(r*Tr*s)
block3 = 1/(1+Tg*s)

c = -block1*block2*(w+c*R)

