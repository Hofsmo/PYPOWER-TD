#!/usr/bin/python3

import control
import matplotlib.pyplot as plt

sys = control.tf([-2, 1], [1, 3])
t, y = control.step_response(sys)

plt.plot(t, y)
plt.show()
