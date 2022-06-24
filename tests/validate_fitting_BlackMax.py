from LifetimeKit.greybody_tables import BlackMax
from matplotlib import pyplot as plt
import numpy as np

astar_grid = np.linspace(0.0, 1.5, num=100)

plt.figure()

plt.plot(astar_grid, BlackMax.xi_dMdt_spin0_spline(astar_grid), label='$dM/dt$ spin0 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dMdt_spin0, label='$dM/dt$ spin0 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dMdt_spin0pt5_spline(astar_grid), label='$dM/dt$ spin1/2 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dMdt_spin0pt5, label='$dM/dt$ spin1/2 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dMdt_spin1_spline(astar_grid), label='$dM/dt$ spin1 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dMdt_spin1, label='$dM/dt$ spin1 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dMdt_spin2_spline(astar_grid), label='$dM/dt$ spin2 fit')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dJdt_spin0_spline(astar_grid), label='$dJ/dt$ spin0 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dJdt_spin0, label='$dJ/dt$ spin0 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dJdt_spin0pt5_spline(astar_grid), label='$dJ/dt$ spin1/2 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dJdt_spin0pt5, label='$dJ/dt$ spin1/2 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dJdt_spin1_spline(astar_grid), label='$dJ/dt$ spin1 fit')
plt.plot(BlackMax._astar, BlackMax._xi_dJdt_spin1, label='$dJ/dt$ spin1 data')
plt.legend()
plt.show()

plt.plot(astar_grid, BlackMax.xi_dJdt_spin2_spline(astar_grid), label='$dJ/dt$ spin2 fit')
plt.legend()
plt.show()
