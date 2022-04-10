import math

from LifetimeKit.constants import RadiationConstant4D, StefanBoltzmannConstant4D, StefanBoltzmannConstant5D, \
    gstar_full_SM
from LifetimeKit.models_base import BaseCosmology
from LifetimeKit.natural_units import M4, Kelvin, Metre, Kilogram, Gram, SolarMass

# numerical constant in test (based on mass M) for whether black hole behaves as 4D Schwarzschild
# or 5D Myers-Perry
Const_4D5D_Transition_Mass = 4.0 / 3.0

# constant in test for whether black hole behaves as 4D Schwarzschild or 5D Myers-Perry
Const_4D5D_Transition_Radius = 2.0 / (3.0 * math.pi)

Const_M_H = 4.0 * math.sqrt(3.0) * math.pi
Const_4Pi = 4.0 * math.pi
Const_2Pi = 2.0 * math.pi
Const_Sqrt_3 = math.sqrt(3.0)
Const_4thRoot_6 = math.pow(6, 1.0 / 4.0)

# coefficients in 4D and 5D black hole mass/radius relations
Const_Radius_5D = 1.0 / (math.sqrt(3.0) * math.pi)
Const_Radius_4D = 1.0 / (4.0 * math.pi)

# effective radius multipliers for 4D and 5D black holes
Const_Reff_5D = 2.0
Const_Reff_4D = 3.0 * math.sqrt(3.0) / 2.0


# Analytic lifetime model for 4D Schwarzschild black hole, for evaporation only, neglecting accretion
def Solve_4D_T(Ti, Mi, Mf, gstar, a, tension, g4, sigma4, g5, sigma5, M4, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = math.sqrt(a*gstar)

    Ti_sq = Ti*Ti
    Ti_4 = Ti_sq*Ti_sq

    tension_sqrt = math.sqrt(tension)

    alpha_sq = alpha*alpha

    M4_sq = M4*M4

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * math.pow(1.0 - Mf_over_Mi*Mf_over_Mi*Mf_over_Mi, 1.0/3.0)
    DeltaM_over_M4 = DeltaM / M4
    DeltaM_over_M4_3 = DeltaM_over_M4 * DeltaM_over_M4 * DeltaM_over_M4

    g_factor = 8.0*g4*sigma4 + g5*alpha*sigma5

    A_const = 64.0*math.sqrt(2.0/3.0)*math.pi/3.0

    A1 = math.sqrt(a_gstar*Ti_4 + 2.0*tension)/Ti_sq
    A2 = A_const * a_gstar_sqrt * tension_sqrt * DeltaM_over_M4_3 / (M4_sq * alpha_sq * g_factor)
    A = A1 + A2

    A_sq = A*A

    return math.pow(2.0*tension / (A_sq - a_gstar), 1.0/4.0)

# Analytic lifetime model for 5D Tangherlini black hole (i.e. Myers-Perry without spin), for evaporation only,
# neglecting accretion
def Solve_5D_T(Ti, Mi, Mf, gstar, a, tension, g4, sigma4, g5, sigma5, M4, M5, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = math.sqrt(a * gstar)

    Ti_sq = Ti*Ti
    Ti_4 = Ti_sq*Ti_sq

    tension_sqrt = math.sqrt(tension)

    alpha_sq = alpha*alpha

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * math.sqrt(1.0 - Mf_over_Mi*Mf_over_Mi)
    DeltaM_over_M5 = DeltaM / M5
    DeltaM_over_M5_sq = DeltaM_over_M5*DeltaM_over_M5

    g_factor = 4.0*g4*sigma4 + g5*alpha*sigma5

    A_const = 16.0*math.sqrt(2.0/3.0)*math.pi/3.0

    A1 = math.sqrt(a_gstar*Ti_4 + 2.0*tension) / Ti_sq
    A2 = A_const * a_gstar_sqrt * tension_sqrt * DeltaM_over_M5_sq / (M4 * M5 * alpha_sq * g_factor)
    A = A1 + A2

    A_sq = A*A

    return math.pow(2.0*tension / (A_sq - a_gstar), 1.0/4.0)


# The *Parameters* class captures details of the 4D and 5D Planck masses, and uses these to compute derived
# quantities such as the brane tension and the crossover temperature (in GeV and Kelvin) from the quadratic
# Hubble regime to the linear regime. It also computes the AdS radius in inverse GeV and metres, assuming that
# the bulk cosmological constant is tuned to produce zero four-dimensional cosmological constant.
# This is important for deciding when the transition takes place from a "small" 5D black hole (approximated
# by Myers-Perry) to a "large", effectively 4D black hole (approximated by 4D Schwarzschild).
class Parameters:

    def __init__(self, M5):
        # M5 is the fundamental 5D *reduced* Planck mass, measured in GeV
        # M4 is the derived 4D *reduced* Planck mass, measured in GeV

        # M5 should not be much less than a TeV
        if M5 < 1E3:
            raise RuntimeError('Parameter error: 5D Planck mass M5 is less than 1 TeV')

        if M4 <= M5:
            raise RuntimeError(
                'Parameter error: 4D Planck mass should be more (usually much more) than the 5D Planck mass')

        # store primary data
        self.M5 = M5
        self.M4 = M4

        # set maximum allowed temperature to be the 5D Planck scale M5
        self.Tmax = M5

        # M_ratio < 1 (usually mu << 1) is the M5/M4 ratio
        M_ratio = M5 / M4
        self.M_ratio = M_ratio

        # compute brane tension lambda = 6 mu^2 M5^4
        # note we can't actually call it lambda in Python, for which lambda is a reserved word
        self.tension = 6 * M_ratio*M_ratio * M5*M5*M5*M5

        # also compute the mass scale associated with the tension
        self.tension_scale = Const_4thRoot_6 * math.sqrt(M_ratio) * M5

        # compute crossover temperature from the quadratic to the linear regime, which occurs when rho = 2 lambda

        # we need the 4D radiation density constant (per degree of freedom) to convert temperature to energy density
        self.RadiationConstant = RadiationConstant4D
        self.StefanBoltzmannConstant4D = StefanBoltzmannConstant4D
        self.StefanBoltzmannConstant5D = StefanBoltzmannConstant5D

        # assume the crossover temperature is high enough that all SM particles are relativistic and in thermal
        # equilibrium, which should be good above Tcross = 200 GeV; we insist Tcross > 1E3 (see below),
        # which is hopefully enough headroom
        self.T_crossover = math.pow(12.0 / (self.RadiationConstant * gstar_full_SM), 1.0/4.0) * math.sqrt(M_ratio) * M5
        self.T_crossover_Kelvin = self.T_crossover / Kelvin

        # compute AdS radius ell and its inverse, mu
        self.mu = M_ratio * M_ratio * M5
        self.ell_AdS = 1.0 / self.mu
        self.ell_AdS_Metres = self.ell_AdS / Metre

        # compute mass-scale at which black holes cross over from 5D to 4D behaviour
        # This is already guaranteed to be larger than the 4D Planck mass scale M4
        self.M_transition = Const_4D5D_Transition_Mass * M5 / (M_ratio*M_ratio*M_ratio*M_ratio)
        self.M_transition_Kilogram = self.M_transition / Kilogram
        self.M_transition_MSun = self.M_transition / SolarMass

        # SANITY CHECKS

        # check that brane tension scale is smaller than 5D Planck scale
        # this is basically implied if M4 > M5, but there are numerical factors that make it
        # worth checking
        if self.tension_scale > M5:
            raise RuntimeError('Parameter error: brane tension lambda = {tension:.3g} GeV should be smaller than '
                               '5D Planck scale {M5:.3g} GeV'.format(tension=self.tension, M5=M5))

        # check that crossover temperature is smaller than 5D Planck scale
        # this is basically implied if M4 > M5, but there are numerical factors that make it
        # worth checking
        if self.T_crossover > M5:
            raise RuntimeError('Parameter error: crossover temperature T_cross = {Tcross:.3g} GeV should be smaller '
                               'than 5D Planck scale {M5:.3g} GeV'.format(Tcross=self.T_crossover, M5=M5))

        # there's no need to compare the tension and crossover temperature to the 4D Planck scale, because
        # we've already guaranteed that the 4D Planck scale is larger than the 5D Planck scale

        # check that crossover temperature is larger than suggested experimental minimum 1E3 GeV from Guedens et al.,
        # see Eq. (18) and discussion below in astro-ph/0205149v2
        if self.T_crossover < 1E3:
            raise RuntimeError('Parameter error: crossover temperature T_cross = {Tcross:.3g} GeV should be larger '
                               'than experimental limit 1E3 GeV = TeV suggested by '
                               'Guedens et al.'.format(Tcross=self.T_crossover))

    def __str__(self):
        return '5D Planck Mass        M5      = {M5:.5g} GeV\n' \
               '4D Planck Mass        M4      = {M4:.5g} GeV\n' \
               '5D/4D mass ratio      M5/M4   = {ratio:.5g}\n' \
               'brane tension         lambda  = {tension:.5g} GeV^4\n' \
               'brane tension scale           = {tscale:.5g} GeV\n' \
               'crossover temperature T_cross = {Tcross:.5g} GeV\n' \
               '                              = {TcrossKelvin:.5g} K\n' \
               '4D/5D transition      BH mass = {Mtransit:.5g} GeV\n' \
               '                              = {MtransitKilogram:.5g} kg\n' \
               '                              = {MtransitMSun:.3g} Msun\n' \
               'AdS curvature length  ell_AdS = {ell:.5g} / GeV\n' \
               '                              = {ellMetre:.5g} m\n' \
               'AdS curvature scale   mu      = {mu:.5g} GeV'.format(M5=self.M5, M4=self.M4, tension=self.tension,
                                                                     tscale=self.tension_scale, Tcross=self.T_crossover,
                                                                     TcrossKelvin=self.T_crossover_Kelvin,
                                                                     Mtransit=self.M_transition,
                                                                     MtransitKilogram=self.M_transition_Kilogram,
                                                                     ell=self.ell_AdS, ellMetre=self.ell_AdS_Metres,
                                                                     MtransitMSun=self.M_transition_MSun,
                                                                     mu=self.mu, ratio=self.M_ratio)


# class *BlackHole* represents the state of a PBH, which involves at least mass but possibly also charge and
# angular momentum. It can be used as an arithmetic type and can be queried for other properties, such as
# the Hawking temperature.
#
# The crossover from 5D to 4D behaviour is taken to occur when the 5D Myers-Perry radius is equal to
# (4/3) * ell, where ell is the AdS curvature radius. This makes the black hole temperatures continuous.
class BlackHole:

    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    # capture (i) initial mass value, and (ii) a RandallSundrumParameters instance so we can decide whether we are in the 4D or
    # 5D regime based on the AdS radius.
    # The initial mass value can be specified in grams, kilograms, or GeV, but defaults to GeV
    def __init__(self, params: Parameters, mass: float, units='GeV'):
        self.params = params

        self._M4_over_M5_sqrt = math.sqrt(1.0/self.params.M_ratio)

        # assign current value
        self.set_value(mass, units)

        # check mass is larger than 4D Planck mass; there's no need to check the 5D Planck mass, because
        # we guarantee that M4 > M5
        if self.mass <= self.params.M4:
            raise RuntimeError('RandallSundrum5D.BlackHole: Initial black hole mass {mass} GeV should be larger than '
                               'the 4D Planck mass {MP} GeV in order that the PBH does not begin life as a '
                               'relic'.format(mass=self.mass, MP=self.params.M4))


    def set_value(self, mass: float, units='GeV'):
        if units not in self._mass_conversions:
            raise RuntimeError('RandallSundrum5D.BlackHole: unit "{unit}" not understood in constructor'.format(unit=units))

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

    # query for the 5D Myers-Perry radius of the black hole, measured in 1/GeV
    @property
    def radius_5D(self):
        return Const_Radius_5D * math.sqrt(self.mass / self.params.M5) / self.params.M5

    # query for the 4D Schwarzschild radius of the black hole, measured in 1/GeV
    @property
    def radius_4D(self):
        return Const_Radius_4D * (self.mass / self.params.M4) / self.params.M4

    # determine whether is black hole is in the 5D or 4D regime
    @property
    def is_5D(self):
        R_5D = self.radius_5D
        if R_5D <= Const_4D5D_Transition_Radius * self.params.ell_AdS:
            return True

        return False

    # query for the radius, measured in 1/GeV, accounting for the 4D to 5D crossover
    @property
    def radius(self):
        if self.is_5D:
            return self.radius_5D

        return self.radius_4D

    # query for 5D effective radius, measured in 1/GeV
    # formula is reff = 2 R_h, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
    @property
    def reff_5D(self):
        return Const_Reff_5D * self.radius_5D

    # query for 4D effective radius, measured in 1/GeV
    # formula is 3 sqrt(3) R_h / 2, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
    # or this is a standard calculation using geometrical capture cross-section arguments
    # https://physics.stackexchange.com/questions/52315/what-is-the-capture-cross-section-of-a-black-hole-region-for-ultra-relativistic
    @property
    def reff_4D(self):
        return Const_Reff_4D * self.radius_4D

    # query for effective radius, measured in 1/GeV, accounting for the 4D to 5D crossover
    @property
    def reff(self):
        if self.is_5D:
            return self.reff_5D

        return self.reff_4D

    # query for correct value of alpha, which determines how the effective radius is related to
    # the horizon radius
    @property
    def alpha(self):
        if self.is_5D:
            return Const_Reff_5D

        return Const_Reff_4D

    # query for the 5D Hawking temperature, measured in GeV
    # the relation is T_H = 1/(4pi R_h)
    @property
    def T_Hawking_5D(self):
        return 1.0 / (Const_2Pi * self.radius_5D)

    # query for the 5D Hawking temperature, measured in GeV
    # the relation is T_H = 1/(2pi R_h)
    @property
    def T_Hawking_4D(self):
        return 1.0 / (Const_4Pi * self.radius_4D)

    # query for the Hawking temperature, measured in GeV, accounting for the 4D to 5D crossover
    @property
    def T_Hawking(self):
        if self.is_5D:
            return self.T_Hawking_5D

        return self.T_Hawking_4D

    # query for t, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
    @property
    def t(self):
        if self.is_5D:
            return Const_2Pi

        return Const_4Pi

    # use an analytic lifetime model to determine the final radiation temperature given a current
    # radiation temperature and a target final mass
    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True):
        # if the PBH is already in the 5D regime, it stays in that regime all the way down to a relic
        if self.is_5D:
            return Solve_5D_T(Ti_rad, self.mass, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                              self.params.tension, 2.0, self.params.StefanBoltzmannConstant4D,
                              5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                              Const_Reff_5D if use_effective_radius else 1.0)

        # otherwise there is a period of 4D evolution, followed by a period of 5D evolution
        T_transition = Solve_4D_T(Ti_rad, self.mass, self.params.M_transition, gstar_full_SM,
                                  self.params.RadiationConstant, self.params.tension,
                                  2.0, self.params.StefanBoltzmannConstant4D,
                                  5.0, self.params.StefanBoltzmannConstant5D, self.params.M4,
                                  Const_Reff_4D if use_effective_radius else 1.0)

        return Solve_5D_T(T_transition, self.params.M_transition, relic_scale, gstar_full_SM,
                          self.params.RadiationConstant, self.params.tension,
                          2.0, self.params.StefanBoltzmannConstant4D,
                          5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                          Const_Reff_5D if use_effective_radius else 1.0)


# The *Model* class provides methods to compute the Hubble rate, Hubble length, horizon mass, etc.,
# for the Randall-Sundrum scenario
class Model(BaseCosmology):

    # allow type introspection for our associated BlackHole model
    BlackHoleType = BlackHole

    def __init__(self, params: Parameters, fixed_g=None):
        super().__init__(params, fixed_g)

    # compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * math.sqrt(rho * (1.0 + rho / (2.0 * self.params.tension)))

    # compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    # the formula here is R_H = 1/H
    def R_Hubble(self, T=None, log_T=None):
        return 1.0 / self.Hubble(T, log_T)

    # compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
    # the formula here is M_H = (4/3) pi rho R_H^3, but we compute it directly to avoid multiple evaluations of rho
    def M_Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 \
              * math.pow(1.0 + rho / (2.0 * self.params.tension), -3.0/2.0) / math.sqrt(rho)

        return M_H
