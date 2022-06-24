from LifetimeKit.greybody_tables import Kerr
from matplotlib import pyplot as plt
import numpy as np

astar_grid = np.linspace(0.0, 1.0, num=100)

plt.figure()

plt.plot(astar_grid, Kerr.xi_dMdt_spin0_spline(astar_grid), label='$dM/dt$ spin0 fit')
plt.plot(Kerr._astar, Kerr._xi_dMdt_spin0, label='$dM/dt$ spin0 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dMdt_spin0pt5_spline(astar_grid), label='$dM/dt$ spin1/2 fit')
plt.plot(Kerr._astar, Kerr._xi_dMdt_spin0pt5, label='$dM/dt$ spin1/2 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dMdt_spin1_spline(astar_grid), label='$dM/dt$ spin1 fit')
plt.plot(Kerr._astar, Kerr._xi_dMdt_spin1, label='$dM/dt$ spin1 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dMdt_spin2_spline(astar_grid), label='$dM/dt$ spin2 fit')
plt.plot(Kerr._astar, Kerr._xi_dMdt_spin2, label='$dM/dt$ spin2 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dJdt_spin0_spline(astar_grid), label='$dJ/dt$ spin0 fit')
plt.plot(Kerr._astar, Kerr._xi_dJdt_spin0, label='$dJ/dt$ spin0 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dJdt_spin0pt5_spline(astar_grid), label='$dJ/dt$ spin1/2 fit')
plt.plot(Kerr._astar, Kerr._xi_dJdt_spin0pt5, label='$dJ/dt$ spin1/2 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dJdt_spin1_spline(astar_grid), label='$dJ/dt$ spin1 fit')
plt.plot(Kerr._astar, Kerr._xi_dJdt_spin1, label='$dJ/dt$ spin1 data')
plt.legend()
plt.show()

plt.plot(astar_grid, Kerr.xi_dJdt_spin2_spline(astar_grid), label='$dJ/dt$ spin2 fit')
plt.plot(Kerr._astar, Kerr._xi_dJdt_spin2, label='$dJ/dt$ spin2 data')
plt.legend()
plt.show()
