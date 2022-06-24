from math import pi, exp

import numpy
import numpy as np
from numpy.polynomial import Chebyshev

from .Friedlander import _xi0_spin0_4D, _xi0_spin0pt5_4D, _xi0_spin1_4D, _xi0_spin2_4D
from ..particle_data import _table_merge, SM_particle_base_table

# taken from Table I of Dong, Kinney & Stojkovic, arXiv:1511.05642v3
_astar = np.asarray([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 0.99, 0.999, 0.99999, 1.0])

# the raw values tabulated by Gong et al. Note these refer to Page f and g factors, and not the xi values that
# we are using
_f0 = np.asarray([7.429E-5,
                  7.442E-5,
                  7.319E-5,
                  7.265E-5,
                  7.097E-5,
                  6.996E-5,
                  7.008E-5,
                  7.119E-5,
                  7.969E-5,
                  1.024E-4,
                  1.551E-4,
                  2.283E-4,
                  2.625E-4,
                  2.667E-4,
                  2.667E-4])

_f0pt5 = np.asarray([8.185E-5,
                     8.343E-5,
                     8.830E-5,
                     9.669E-5,
                     1.089E-4,
                     1.258E-4,
                     1.487E-4,
                     1.804E-4,
                     2.284E-4,
                     3.195E-4,
                     4.567E-4,
                     6.708E-4,
                     9.253E-4,
                     1.074E-3,
                     1.093E-3])

_f1 = np.asarray([3.366E-5,
                  3.580E-5,
                  4.265E-5,
                  5.525E-5,
                  7.570E-5,
                  1.080E-4,
                  1.594E-4,
                  2.450E-4,
                  4.014E-4,
                  7.520E-4,
                  1.313E-3,
                  2.151E-3,
                  3.057E-3,
                  3.555E-3,
                  3.616E-3])

_f2 = np.asarray([3.845E-6,
                  4.684E-6,
                  7.732E-6,
                  1.494E-5,
                  3.116E-5,
                  6.822E-5,
                  1.574E-4,
                  3.909E-4,
                  1.104E-3,
                  4.107E-3,
                  1.305E-2,
                  3.578E-2,
                  7.251E-2,
                  9.785E-2,
                  1.012E-1])

_g0 = np.asarray([8.867E-5,
                  9.085E-5,
                  9.391E-5,
                  1.024E-4,
                  1.125E-4,
                  1.281E-4,
                  1.507E-4,
                  1.803E-4,
                  2.306E-4,
                  3.166E-4,
                  4.515E-4,
                  6.160E-4,
                  6.905E-4,
                  6.997E-4,
                  7.006E-4])

_g0pt5 = np.asarray([6.161E-4,
                     6.174E-4,
                     6.218E-4,
                     6.299E-4,
                     6.430E-4,
                     6.631E-4,
                     6.946E-4,
                     7.457E-4,
                     8.366E-4,
                     1.034E-3,
                     1.343E-3,
                     1.810E-3,
                     2.340E-3,
                     2.641E-3,
                     2.678E-3])

_g1 = np.asarray([4.795E-4,
                  4.895E-4,
                  5.207E-4,
                  5.759E-4,
                  6.599E-4,
                  7.845E-4,
                  9.668E-4,
                  1.245E-3,
                  1.706E-3,
                  2.636E-3,
                  3.976E-3,
                  5.829E-3,
                  7.723E-3,
                  8.730E-3,
                  8.851E-3])

_g2 = np.asarray([1.064E-4,
                  1.167E-4,
                  1.514E-4,
                  2.233E-4,
                  3.603E-4,
                  6.236E-4,
                  1.155E-3,
                  2.322E-3,
                  5.286E-3,
                  1.544E-2,
                  4.057E-2,
                  9.555E-2,
                  1.753E-1,
                  2.271E-1,
                  2.338E-1])

# convert Page f factors to xi factors
# for details, see 15 June 2022 calculation "Relationship between Page f, g factors and Friedlander et al. xi factors"
def _convert_f_to_xi(astar_sample, f_sample):
    astar_sq = astar_sample * astar_sample
    return 2.0 * pi * (2.0 - astar_sq + 2.0*np.sqrt(1.0 - astar_sq)) * f_sample

_xi_dMdt_spin0 = _convert_f_to_xi(_astar, _f0)
_xi_dMdt_spin0pt5 = _convert_f_to_xi(_astar, _f0pt5)
_xi_dMdt_spin1 = _convert_f_to_xi(_astar, _f1)
_xi_dMdt_spin2 = _convert_f_to_xi(_astar, _f2)

# convert Page g factors to xi factors
# for details, see same calcuation
def _convert_g_to_xi(astar_sample, g_sample):
    astar_sq = astar_sample * astar_sample
    return 2.0 * pi * (1.0 + np.sqrt(1.0 - astar_sq)) * astar_sample * g_sample

_xi_dJdt_spin0 = _convert_g_to_xi(_astar, _g0)
_xi_dJdt_spin0pt5 = _convert_g_to_xi(_astar, _g0pt5)
_xi_dJdt_spin1 = _convert_g_to_xi(_astar, _g1)
_xi_dJdt_spin2 = _convert_g_to_xi(_astar, _g2)

class _Kerr_interpolator:
    """
    An interpolating class for the Kerr emission rates. If we carelessly fit a spline to these functions,
    there tend to be wild oscillations – so some circumspection is needed
    """

    def __init__(self, astar_grid, value_grid, value_at_zero, deg=3) -> None:
        """
        Capture the (a*, value) sample grid. Note that the astar grid usually does not begin at a* = 0,
        so we have a separate 'value_at_zero' parameter. We linearly interpolate between
        value_at_zero and the first entry in value_grid for small values of astar
        :param astar_grid:
        :param value_grid:
        :param value_at_zero:
        """

        self._astar_grid = np.asarray(astar_grid)

        self._value_grid = np.asarray(value_grid)
        self._log_value_grid = np.log(self._value_grid)

        self._cfit = Chebyshev.fit(self._astar_grid, self._log_value_grid, deg=deg)

        self._value_at_zero = value_at_zero
        self._first_astar = self._astar_grid[0]
        self._first_value = self._cfit(self._first_astar)   # ensure the fitting function is cts

    def __call__(self, astar):
        if isinstance(astar, float):
            return self._evaluate(astar)

        if isinstance(astar, list):
            return [self._evaluate(a) for a in astar]

        if isinstance(astar, numpy.ndarray):
            return np.array(list(map(self._evaluate, astar)), dtype=astar.dtype)

        return self._evaluate(astar)


    def _evaluate(self, astar: float) -> float:
        if astar < 0.0:
            raise ValueError('Kerr a* parameter should be >= 0.0')
        if astar > 1.0:
            raise ValueError('Kerr a* parameter should be <= 1.0')

        if astar <= self._first_astar:
            # linearly interpolate
            return self._value_at_zero + (astar / self._first_astar) * (self._first_astar - self._value_at_zero)

        return exp(self._cfit(astar))


# build splines for each of these functions
xi_dMdt_spin0_spline = _Kerr_interpolator(_astar, _xi_dMdt_spin0, _xi0_spin0_4D, deg=5)
xi_dMdt_spin0pt5_spline = _Kerr_interpolator(_astar, _xi_dMdt_spin0pt5, _xi0_spin0pt5_4D, deg=9)
xi_dMdt_spin1_spline = _Kerr_interpolator(_astar, _xi_dMdt_spin1, _xi0_spin1_4D, deg=5)
xi_dMdt_spin2_spline = _Kerr_interpolator(_astar, _xi_dMdt_spin2, _xi0_spin2_4D, deg=5)

xi_dJdt_spin0_spline = _Kerr_interpolator(_astar, _xi_dJdt_spin0, 0.0, deg=7)
xi_dJdt_spin0pt5_spline = _Kerr_interpolator(_astar, _xi_dJdt_spin0pt5, 0.0, deg=7)
xi_dJdt_spin1_spline = _Kerr_interpolator(_astar, _xi_dJdt_spin1, 0.0, deg=7)
xi_dJdt_spin2_spline = _Kerr_interpolator(_astar, _xi_dJdt_spin2, 0.0, deg=7)

Kerr_greybody_table_4D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi_M': xi_dMdt_spin0_spline, 'xi_J': xi_dJdt_spin0_spline, 'xi-per-dof': True},
  'photon': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'gluon': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'W boson': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'Z boson': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'neutrino': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'electron': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'muon': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'tau': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'up quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'down quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'strange quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'charm quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'bottom quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'top quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True}})

# there seems to be no explicit statement in the Dong et al. paper, but to matcher the Friedlander et al.
# xi values we must apparently assume the quoted values for f and g in the spin-2 case already have the
# number of polarizations included
Kerr_graviton_greybody_table_4D = \
 {'4D graviton': {'mass': 0.0, 'dof': 2.0,
                  'xi_M': xi_dMdt_spin2_spline, 'xi_J': xi_dJdt_spin2_spline, 'xi-per-dof': False}}
