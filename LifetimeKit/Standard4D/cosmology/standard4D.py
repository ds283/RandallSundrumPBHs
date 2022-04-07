import numpy as np

from ...models_base import BaseCosmology
from ...natural_units import Kilogram, Gram
from ...constants import gstar_full_SM

Const_M_H = 4.0 * np.sqrt(3.0) * np.pi
Const_Sqrt_3 = np.sqrt(3.0)
Const_4Pi = 4.0 * np.pi

Const_Radius_4D = 1.0 / (4.0 * np.pi)
Const_Reff_4D = 3.0 * np.sqrt(3.0) / 2.0


# Analytic lifetime model for 4D Schwarzschild black hole, for evaporation only, neglecting accretion
def Solve_4D_T(Ti, Mi, Mf, gstar, a, g4, sigma4, M4, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = np.sqrt(a*gstar)

    Ti_sq = Ti*Ti
    Ti_4 = Ti_sq*Ti_sq

    alpha_sq = alpha*alpha

    M4_sq = M4*M4

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * np.power(1.0 - Mf_over_Mi*Mf_over_Mi*Mf_over_Mi, 1.0/3.0)
    DeltaM_over_M4 = DeltaM / M4
    DeltaM_over_M4_3 = DeltaM_over_M4 * DeltaM_over_M4 * DeltaM_over_M4

    g_factor = g4*sigma4

    A_const = (8.0/3.0*np.sqrt(3.0))*np.pi

    A1 = 1.0/Ti_sq
    A2 = A_const * a_gstar_sqrt * DeltaM_over_M4_3 / (M4_sq * alpha_sq * g_factor)
    A = A1 + A2

    return 1.0/np.sqrt(A)


# class *BlackHole* represents the state of a PBH, which involves at least mass but possibly also charge and
# angular momentum. It can be used as an arithmetic type and can be queried for other properties, such as
# the Hawking temperature.
class BlackHole:

    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    # capture (i) initial mass value, and (ii) a RandallSundrumParameters instance so we can decide whether we are in the 4D or
    # 5D regime based on the AdS radius.
    # The initial mass value can be specified in grams, kilograms, or GeV, but defaults to GeV
    def __init__(self, params, mass: float, units='GeV'):
        self.params = params

        # assign current value
        self.set_value(mass, units)

        # check mass is larger than 4D Planck mass
        if self.mass <= self.params.M4:
            raise RuntimeError('Standard4D.BlackHole: Initial black hole mass {mass} GeV should be larger than the '
                               '4D Planck mass {MP} GeV in order that the PBH does not begin life as a '
                               'relic'.format(mass=self.mass, MP=self.params.M4))


    def set_value(self, mass: float, units='GeV'):
        if units not in self._mass_conversions:
            raise RuntimeError('Standard4D.BlackHole: unit "{unit}" not understood in constructor'.format(unit=units))

        units_to_GeV = self._mass_conversions[units]
        self.mass = mass * units_to_GeV


    # implement basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, BlackHole):
            return BlackHole(self.params, self.mass + other.mass)

        return BlackHole(self.params, self.mass + other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, BlackHole):
            return BlackHole(self.params, self.mass - other.mass)

        return BlackHole(self.params, self.mass - other)

    def __rsub__(self, other):
        if isinstance(other, BlackHole):
            return BlackHole(self.params, other.mass - self.mass)

        return BlackHole(self.params, other - self.mass)

    def __mul__(self, other):
        if isinstance(other, BlackHole):
            return NotImplemented

        return BlackHole(self.params, other * self.mass)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, BlackHole):
            return NotImplemented

        return BlackHole(self.params, self.mass / other)

    # query for the 4D Schwarzschild radius of the black hole, measured in 1/GeV
    @property
    def radius(self):
        return Const_Radius_4D * (self.mass / self.params.M4) / self.params.M4

    # query for 4D effective radius, measured in 1/GeV
    # formula is 3 sqrt(3) R_h / 2, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
    # or this is a standard calculation using geometrical capture cross-section arguments
    # https://physics.stackexchange.com/questions/52315/what-is-the-capture-cross-section-of-a-black-hole-region-for-ultra-relativistic
    @property
    def reff(self):
        return Const_Reff_4D * self.radius

    # query for correct value of alpha, which determines how the effective radius is related to
    # the horizon radius
    @property
    def alpha(self):
        return Const_Reff_4D

    # query for the 5D Hawking temperature, measured in GeV
    # the relation is T_H = 1/(2pi R_h)
    @property
    def T_Hawking(self):
        return 1.0 / (Const_4Pi * self.radius)

    # query for t, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
    @property
    def t(self):
        return Const_4Pi

    # use an analytic lifetime model to determine the final radiation temperature given a current
    # radiation temperature and a target final mass
    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True):
        return Solve_4D_T(Ti_rad, self.mass, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                          2.0, self.params.StefanBoltzmannConstant4D, self.params.M4,
                          Const_Reff_4D if use_effective_radius else 1.0)


# The *Model* class provides methods to compute the Hubble rate, Hubble length, horizon mass, etc.,
# for the Randall-Sundrum scenario
class Model(BaseCosmology):

    # allow type introspection for our associated BlackHole model
    BlackHoleType = BlackHole

    def __init__(self, params, fixed_g=None):
        super().__init__(params, fixed_g)

    # compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * np.sqrt(rho)

    # compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    # the formula here is R_H = 1/H
    def R_Hubble(self, T=None, log_T=None):
        return 1.0 / self.Hubble(T, log_T)

    # compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
    def M_Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 / np.sqrt(rho)

        return M_H
