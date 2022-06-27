import math

from LifetimeKit.constants import RadiationConstant4D, StefanBoltzmannConstant4D, StefanBoltzmannConstant5D, \
    gstar_full_SM
from LifetimeKit.models_base import BaseCosmology, BaseBlackHole, BaseSpinningBlackHole
from LifetimeKit.natural_units import M4, Kelvin, Metre, Kilogram, Gram, SolarMass

# numerical constant in test (based on mass M) for whether black hole behaves as 4D Schwarzschild
# or 5D Myers-Perry
Const_4D5D_Transition_Mass = 4.0 / 3.0

# constant in test for whether black hole behaves as 4D Schwarzschild or 5D Myers-Perry
Const_4D5D_Transition_Radius = 2.0 / (3.0 * math.pi)

Const_M_H = 4.0 * math.sqrt(3.0) * math.pi
Const_8Pi = 8.0 * math.pi
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

# constant appearing in crossover mass where Kerr and Myers-Perry limits on J meet
Const_J_crossover = 256.0/17.0

# constant appearing in calculation of J limit for Myers-Perry
Const_J_limit_Myers_Perry = 2.0 / (3.0 * math.sqrt(3.0) * math.pi)

# constant appearing in calculation of the Myers-Perry my parameter
Const_mu_prefactor = 3.0 * math.pi * math.pi

_ASTAR_TOLERANCE = 1E-6


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

class SpinlessBlackHole(BaseBlackHole):
    """
    Represents a Schwarzschild-like black hole on a Randall-Sundrum brane. We model this using a Schwarzschild metric
    when the black hole radius is large (it must be squashed in the extra dimension, so look like a
    Scharzschild-like pancake when viewed from the bulk; an exact analytic metric is lacking),
    and a 5D Myers-Perry black hole (without spin, i.e., the 5D Tangherlini metric) when the black hole
    radius is small (then it cannot feel the constraints from the extra dimension).

    The crossover from 5D to 4D behaviour is taken to occur when the 5D Myers-Perry radius is equal to
    (4/3) * ell, where ell is the AdS curvature radius. This makes the black hole temperatures continuous.
    """

    def __init__(self, params: Parameters, M: float, units='GeV', strict=True) -> None:
        """
        capture (i) initial mass value, and (ii) a parameter container instance so we can decide whether
        we are in the 4D or 5D regime based on the AdS radius.
        The initial mass value can be specified in grams, kilograms, or GeV, but defaults to GeV
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param units: units used to measure the black hole mass
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__(params, M, units=units, strict=strict)

        # cache value of sqrt(M4/M5)
        self._M4_over_M5_sqrt = math.sqrt(1.0/self.params.M_ratio)


    @property
    def radius_5D(self) -> float:
        """
        query for the 5D Myers-Perry radius of the black hole, measured in 1/GeV
        """
        return Const_Radius_5D * math.sqrt(self.M / self.params.M5) / self.params.M5

    @property
    def radius_4D(self) -> float:
        """
        query for the 4D Schwarzschild radius of the black hole, measured in 1/GeV
        :return:
        """
        return Const_Radius_4D * (self.M / self.params.M4) / self.params.M4

    @property
    def is_5D(self) -> bool:
        """
        determine whether is black hole is in the 5D or 4D regime
        :return:
        """
        R_5D = self.radius_5D
        if R_5D <= Const_4D5D_Transition_Radius * self.params.ell_AdS:
            return True

        return False

    @property
    def radius(self) -> float:
        """
        determine radius of the black hole, accounting for 4D -> 5D transitions
        :return:
        """
        if self.is_5D:
            return self.radius_5D

        return self.radius_4D

    @property
    def reff_5D(self) -> float:
        """
        query for 5D effective radius, measured in 1/GeV
        formula is reff = 2 R_h, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
        :return:
        """
        return Const_Reff_5D * self.radius_5D

    @property
    def reff_4D(self) -> float:
        """
        query for 4D effective radius, measured in 1/GeV
        formula is 3 sqrt(3) R_h / 2, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
        or this is a standard calculation using geometrical capture cross-section arguments
        https://physics.stackexchange.com/questions/52315/what-is-the-capture-cross-section-of-a-black-hole-region-for-ultra-relativistic
        :return:
        """
        return Const_Reff_4D * self.radius_4D

    @property
    def reff(self) -> float:
        """
        query for effective radius, measured in 1/GeV, accounting for the 4D to 5D crossover
        :return:
        """
        if self.is_5D:
            return self.reff_5D

        return self.reff_4D

    @property
    def alpha(self) -> float:
        """
        query for correct value of alpha, which determines how the effective radius is related to
        the horizon radius
        :return:
        """
        if self.is_5D:
            return Const_Reff_5D

        return Const_Reff_4D

    @property
    def T_Hawking_5D(self) -> float:
        """
        query for the 5D Hawking temperature, measured in GeV
        the relation is T_H = 1/(4pi R_h)
        :return:
        """
        try:
            return 1.0 / (Const_2Pi * self.radius_5D)
        except ZeroDivisionError:
            pass

        return float("nan")

    @property
    def T_Hawking_4D(self) -> float:
        """
        query for the 5D Hawking temperature, measured in GeV
        the relation is T_H = 1/(2pi R_h)
        :return:
        """
        try:
            return 1.0 / (Const_4Pi * self.radius_4D)
        except ZeroDivisionError:
            pass

        return float("nan")

    @property
    def T_Hawking(self) -> float:
        """
        query for the Hawking temperature, measured in GeV, accounting for the 4D to 5D crossover
        :return:
        """
        if self.is_5D:
            return self.T_Hawking_5D

        return self.T_Hawking_4D

    @property
    def t(self) -> float:
        """
        query for t, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
        :return:
        """
        if self.is_5D:
            return Const_2Pi

        return Const_4Pi

    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True) -> float:
        """
        use an analytic lifetime model to determine the final radiation temperature given a current
        radiation temperature and a target final mass
        :param Ti_rad:
        :param relic_scale:
        :param use_effective_radius:
        :return:
        """
        # if the PBH is already in the 5D regime, it stays in that regime all the way down to a relic
        if self.is_5D:
            return Solve_5D_T(Ti_rad, self.M, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                              self.params.tension, 2.0, self.params.StefanBoltzmannConstant4D,
                              5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                              Const_Reff_5D if use_effective_radius else 1.0)

        # otherwise there is a period of 4D evolution, followed by a period of 5D evolution
        T_transition = Solve_4D_T(Ti_rad, self.M, self.params.M_transition, gstar_full_SM,
                                  self.params.RadiationConstant, self.params.tension,
                                  2.0, self.params.StefanBoltzmannConstant4D,
                                  5.0, self.params.StefanBoltzmannConstant5D, self.params.M4,
                                  Const_Reff_4D if use_effective_radius else 1.0)

        return Solve_5D_T(T_transition, self.params.M_transition, relic_scale, gstar_full_SM,
                          self.params.RadiationConstant, self.params.tension,
                          2.0, self.params.StefanBoltzmannConstant4D,
                          5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                          Const_Reff_5D if use_effective_radius else 1.0)

class SpinningBlackHole(BaseSpinningBlackHole):
    """
    Represents a spinning black hole on a Randall-Sundrum brane. We model this using a Kerr metric
    when the black hole radius is large, and a 5D Myers-Perry black hole with single spin parameter when
    the black hole radius is small.

    The crosover from 5D to 4D behaviour is taken to occur when the 5D Myers-Perry radius is equal to the
    curvature radius of the extra dimenion ell. This is slightly different to the prescription used in
    SpinlesBlackHole, which just represents the ambiguity of when the transition is taken to occur.
    In SpinlessBlackHole the prescription is chosen so that the black hole temperature is continuous.
    Here, it has to be chosen as a measure that includes the black hole spin, since that can affect
    whether the hole is behaving in its 5D or 4D regime.
    """

    def __init__(self, params, M: float, J: float=None, J_over_Jmax: float=None, units='GeV', strict=True):
        """
        Instantiate a brane black hole model with spin. This requires a specification of the mass M
        and angular momentum J associated with the black hole. Note that we don't allow pec
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param J: (optional) black hole angular momentum, which is dimensionless
        :param J_over_Jmax: black hole angular momentum, specified in terms of J/Jmax. Here Jmax is always
        theh Myers-Perry limit on the spin parameter, because that is always the controlling one.
        If both are specified, J/Jmax is used in preference. If neither is specified, the angular momentum is set
        to zero.
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__(params, M, units=units, strict=strict)

        self._M4_over_M5 = self.params.M4 / self.params.M5
        self._M4_over_M5_cube = self._M4_over_M5*self._M4_over_M5*self._M4_over_M5
        self._J_crossover_scale = Const_J_crossover * self._M4_over_M5_cube * self.params.M4

        self._mu_prefactor = Const_mu_prefactor * self.params.M5*self.params.M5*self.params.M5

        # assign current angular momentum value
        # define a 'None' value first, in order to define all instance attributes within __init__()
        self.J = None
        self.set_J(J=J, J_over_Jmax=J_over_Jmax)

    @property
    def J_limit_5D(self) -> float:
        """
        query for the maximum allowed J for the current value of M, interpreting this is a 5D
        Myers-Perry black hole
        :return:
        """
        M_M5 = self.M / self.params.M5
        M_M5_power32 = math.pow(M_M5, 3.0 / 2.0)
        return Const_J_limit_Myers_Perry * M_M5_power32

    @property
    def J_limit_4D(self) -> float:
        """
        query for the maximum allowed J for the current value of M, interpreting this as a 4D
        Kerr black hole
        :return:
        """
        M_M4 = self.M / self.params.M4
        M_M4_sq = M_M4 * M_M4
        return M_M4_sq / Const_8Pi

    @property
    def J_limit(self) -> float:
        """
        query for the maximum currently-allowed J. For a Randall-Sundrum 5D black hole, that is always the
        Myers-Perry limit, because: (i) at low M, the black hole is always in its Myers-Perry phase anyway, and
        the corresponding J_max is the appropriate one; and (ii) at higher M, if we increase J, the black hole will
        do a dimensional transition and get to a Myers-Perry phase before J can approach the Kerr J_max.
        :return:
        """
        return self.J_limit_5D

    def set_J(self, J: float=None, J_over_Jmax: float=None) -> None:
        """
        Sets the current value of J for this PBH, either by specifying it as the direct spin parameter J,
        or (probably more convenient) as the ratio J/Jmax.
        Note that for a Randall-Sundrum black hole, the Jmax for Myers-Perry is always the relevant one.
        :param J:
        :param J_over_Jmax:
        :return:
        """
        J_limit = self.J_limit

        if J_over_Jmax is not None:
            if J_over_Jmax < 0.0:
                raise RuntimeError('RandallSundrum5D.SpinningBlackHole: angular momentum parameter J/Jmax should not '
                                   'be negative (requested value was J/Jmax={JJmax})'.format(JJmax=J_over_Jmax))
            if J_over_Jmax > 1.0:
                raise RuntimeError('RandallSundrum5D.SpinningBlackHole: angular momentum parameter J/Jmax should be '
                                   'less than unity (requested value was J/Jmax={JJmax})'.format(JJmax=J_over_Jmax))

            # the 5D value of Jmax is always controlling,
            self.J = J_over_Jmax * J_limit

        elif J is not None:
            # check supplied value of J is valid
            if J < 0.0:
                raise RuntimeError('RandallSundrum5D.SpinningBlackHole: angular momentum J should not be negative '
                                   '(requested value was J={J})'.format(J=J))

            if J > J_limit:
                raise RuntimeError('RandallSundrum5D.SpinningBlackHole: angular momentum J exceeds maximum allowed value Jmax={Jmax} '
                                   '(requested value was J={J})'.format(Jmax=J_limit, J=J))
            self.J = J

        else:
            raise RuntimeError('RandallSundrum5D.SpinningBlackHole: neither J nor J/Jmax was specified')

    @property
    def mu(self) -> float:
        """
        query for Myers-Perry mu mass parameter
        :return:
        """
        return self.M / self._mu_prefactor

    @property
    def a(self) -> float:
        """
        query for the Myers-Perry angular momentum parameter j
        :return:
        """
        return (3.0*self.J) / (2.0*self.M)

    @property
    def radius_5D(self) -> float:
        """
        query for current radius of the black hole horizon computed using 5D Myers-Perry formula,
        measured in 1/GeV
        formula is R_h = sqrt(mu - a^2) where mu is the Myers-Perry mass parameter
        :return:
        """
        a_sq = self.a * self.a

        if (a_sq - self.mu)/self.mu > _ASTAR_TOLERANCE:
            raise ValueError('RandallSundrum5D.SpinningBlackHole: radius_5D angular momentum value too large '
                             '(J = {J}, Jmax_5D = {Jmax_5D}, Jmax_4D = {Jmax_4D}, J/Jmax_5D = {Jratio_5D}, J/Jmax_4D '
                             '= {Jratio_4D})'.format(J=self.J, Jmax_5D=self.J_limit_5D, Jmax_4D=self.J_limit_4D,
                                                     Jratio_5D=self.J/self.J_limit_5D, Jratio_4D=self.J/self.J_limit_4D))

        if a_sq > self.mu:
            return 0.0

        return math.sqrt(self.mu - a_sq)

    @property
    def radius_4D(self) -> float:
        """
        query for the current radius of the black hole horizon, computed using the Kerr formula,
        measured in 1/GeV
        the formula is R_h = (R_s/2) * (1 + sqrt(1-astar^2)) where R_s = 2MG is the Schwarzschild radius
        :return:
        """
        astar = self.astar_Kerr

        # a* > 1 should be impossible, because we will arrive at an extremal Myers-Perry black hole before we
        # get there ... but test, just to be sure
        if astar - 1.0 > _ASTAR_TOLERANCE:
            raise ValueError('RandallSundrum5D.SpinningBlackHole: radius_4D angular momentum value too large '
                             '(J = {J}, Jmax_5D = {Jmax_5D}, Jmax_4D = {Jmax_4D}, J/Jmax_5D = {Jratio_5D}, J/Jmax_4D '
                             '= {Jratio_4D})'.format(J=self.J, Jmax_5D=self.J_limit_5D, Jmax_4D=self.J_limit_4D,
                                                     Jratio_5D=self.J/self.J_limit_5D, Jratio_4D=self.J/self.J_limit_4D))

        Rs = Const_Radius_4D * (self.M / self.params.M4) / self.params.M4

        if astar > 1.0:
            return Rs / 2.0

        astar_sq = astar*astar
        return (Rs / 2.0) * (1.0 + math.sqrt(1.0 - astar_sq))

    @property
    def is_5D(self) -> bool:
        """
        determine whether is black hole is in the 5D or 4D regime
        :return:
        """
        R_5D = self.radius_5D
        if R_5D <= self.params.ell_AdS:
            return True

        return False

    @property
    def radius(self) -> float:
        """
        determine radius of the black hole, accounting for 4D -> 5D transitions
        :return:
        """
        if self.is_5D:
            return self.radius_5D

        return self.radius_4D

    @property
    def astar_Kerr(self) -> float:
        """
        query for the current value of astar computed using the Kerr formula
        """
        return self.J/self.J_limit_4D

    @property
    def astar_MyersPerry(self) -> float:
        """
        query for the current value of astar, computed using the Myers-Perr formula astar = a/rh
        :return:
        """
        return self.a/self.radius_5D

    @property
    def xi_astar_argument(self) -> float:
        """
        query for the current value of the parameter a* that is needed as an argument for the xi(a*) fitting functions.
        This should be the Kerr a* parameter a* = J/Jmax when the black hole is in its Kerr phase, and
        the Myers-Perry a* parameter a* = a/Rh when the black hole is in its Myers-Perry phase.
        :return:
        """
        if self.is_5D:
            return self.astar_MyersPerry

        return self.astar_Kerr

    @property
    def J_over_Jmax(self):
        """
        query for the current value of J/Jmax. This is used in preference to a*, which has an ambiguous
        interpretation - it isn't defined in the same way for Kerr (where J = a* J_max) and Myers-Perry
        (where J = (a*/sqrt(1+a*^2)) J_max)
        :return:
        """
        return self.J / self.J_limit_5D

    @property
    def reff_5D(self) -> float:
        """
        query for 5D effective radius, measured in 1/GeV
        formula is reff = 2 R_h, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
        TODO: It's not clear the J=0 value is the right answer here - may need to check the literature,
         or just accept that it's a bodge
        :return:
        """
        return Const_Reff_5D * self.radius_5D
    @property
    def reff_4D(self) -> float:
        """
        query for 4D effective radius, measured in 1/GeV
        formula is 3 sqrt(3) R_h / 2, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
        or this is a standard calculation using geometrical capture cross-section arguments
        https://physics.stackexchange.com/questions/52315/what-is-the-capture-cross-section-of-a-black-hole-region-for-ultra-relativistic
        TODO: It's not clear the J=0 value is the right answer here - may need to check the literature,
         or just accept that it's a bodge
        :return:
        """
        return Const_Reff_4D * self.radius_4D
    @property
    def reff(self) -> float:
        """
        query for effective radius, measured in 1/GeV, accounting for the 4D to 5D crossover
        :return:
        """
        if self.is_5D:
            return self.reff_5D

        return self.reff_4D
    @property
    def alpha(self) -> float:
        """
        query for correct value of alpha, which determines how the effective radius is related to
        the horizon radius
        :return:
        """
        if self.is_5D:
            return Const_Reff_5D

        return Const_Reff_4D

    @property
    def T_Hawking_5D(self) -> float:
        """
        query for the 5D Myers-Perry Hawking temperature, measured in GeV
        :return:
        """
        try:
            astar = self.astar_MyersPerry
            astar_sq = astar*astar
            return 1.0 / (Const_2Pi * self.radius_5D * math.sqrt(1 + astar_sq))
        except ZeroDivisionError:
            pass

        return float("nan")

    @property
    def T_Hawking_4D(self) -> float:
        """
        query for the 4D Kerr Hawking temperature, mneasured in GeV
        :return:
        """
        astar = self.astar_Kerr
        try:
            return 1.0 / (Const_4Pi * self.radius_4D) * math.sqrt(1.0 - astar*astar)
        except ZeroDivisionError:
            pass

        return float("nan")
    @property
    def T_Hawking(self):
        """
        query for the Hawking temperature, measured in GeV, accounting for the 4D to 5D crossover
        :return:
        """
        if self.is_5D:
            return self.T_Hawking_5D

        return self.T_Hawking_4D

    @property
    def t_4D(self) -> float:
        """
        query for the Kerr t parameter, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
        """
        astar = self.astar_Kerr
        return Const_4Pi / math.sqrt(1.0 - astar*astar)

    @property
    def t_5D(self) -> float:
        """
        query for the Myers-Perry t parameter
        :return:
        """
        astar = self.astar_MyersPerry
        astar_sq = astar*astar
        return Const_2Pi * math.sqrt(1 + astar_sq)
    @property
    def t(self):
        """
        query for the t parameter, accounting for the 4D to 5D crossover
        :return:
        """
        if self.is_5D:
            return self.t_5D

        return self.t_4D
    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True) -> float:
        """
        use an analytic lifetime model to determine the final radiation temperature given a current
        radiation temperature and a target final mass
        TODO: does not account for angular momentum! hopefully not important
        :param Ti_rad:
        :param relic_scale:
        :param use_effective_radius:
        :return:
        """
        # TODO: assume evaporation logic is not modified by spin (can there by multple 4D -> 5D transitions
        #  or vice versa? Should check)
        # if the PBH is already in the 5D regime, it stays in that regime all the way down to a relic
        if self.is_5D:
            return Solve_5D_T(Ti_rad, self.M, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                              self.params.tension, 2.0, self.params.StefanBoltzmannConstant4D,
                              5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                              Const_Reff_5D if use_effective_radius else 1.0)

        # otherwise there is a period of 4D evolution, followed by a period of 5D evolution
        T_transition = Solve_4D_T(Ti_rad, self.M, self.params.M_transition, gstar_full_SM,
                                  self.params.RadiationConstant, self.params.tension,
                                  2.0, self.params.StefanBoltzmannConstant4D,
                                  5.0, self.params.StefanBoltzmannConstant5D, self.params.M4,
                                  Const_Reff_4D if use_effective_radius else 1.0)

        return Solve_5D_T(T_transition, self.params.M_transition, relic_scale, gstar_full_SM,
                          self.params.RadiationConstant, self.params.tension,
                          2.0, self.params.StefanBoltzmannConstant4D,
                          5.0, self.params.StefanBoltzmannConstant5D, self.params.M4, self.params.M5,
                          Const_Reff_5D if use_effective_radius else 1.0)


class Model(BaseCosmology):
    """
    The *Model* class provides methods to compute the Hubble rate, Hubble length, horizon mass, etc.,
    for the Randall-Sundrum scenario
    """

    def __init__(self, params: Parameters, fixed_g=None):
        super().__init__(params, fixed_g)

    def Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV

        :param T:
        :param log_T:
        :return:
        """
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self._params.M4) * math.sqrt(rho * (1.0 + rho / (2.0 * self._params.tension)))

    def R_Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
        the formula here is R_H = 1/H
        :param T:
        :param log_T:
        :return:
        """
        return 1.0 / self.Hubble(T, log_T)

    def M_Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
        the formula here is M_H = (4/3) pi rho R_H^3, but we compute it directly to avoid multiple evaluations of rho
        :param T:
        :param log_T:
        :return:
        """
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self._params.M4 * self._params.M4 * self._params.M4 \
              * math.pow(1.0 + rho / (2.0 * self._params.tension), -3.0 / 2.0) / math.sqrt(rho)

        return M_H
