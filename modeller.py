import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# introduce fixed constants
Const_8Pi = 8.0 * np.pi
Const_4Pi = 4.0 * np.pi
Const_2Pi = 2.0 * np.pi
Const_PiOver2 = np.pi / 2.0
Const_2PiSquared = 2.0 * np.pi*np.pi
Const_4thRoot_6 = np.power(6, 1.0 / 4.0)
Const_Sqrt_3 = np.sqrt(3.0)
Const_M_H = 4.0 * np.sqrt(3.0) * np.pi
Const_Radius_5D = 1.0 / (np.sqrt(3.0) * np.pi)
Const_Radius_4D = 1.0 / (4.0 * np.pi)
Const_4D5D_Transition_Mass = 4.0 / 3.0
Const_4D5D_Transition_Radius = 2.0 / (3.0 * np.pi)
Const_Reff_5D = 2.0
Const_Reff_4D = 3.0 * np.sqrt(3.0) / 2.0
Const_Hrh_5D = 2.0 / (np.power(3.0, 3.0/4.0) * np.sqrt(np.pi))

RadiationConstant4D = np.pi * np.pi / 30.0
StefanBoltzmannConstant4D = RadiationConstant4D / 4.0
StefanBoltzmannConstant5D = 0.0668850223995 # 2 * Zeta(5) / pi^3
gstar_full_SM = 106.7
T_CMB = 2.72548

# number of points to use on plots
NumPoints = 200

# number of T-sample points to capture for PBH lifetime mass/temperature relation
NumTSamplePoints = 200


# CONVERSION FROM NATURAL UNITS TO SI UNITS

# 4D reduced Planck mass measured in GeV
M4 = 2.435E18

# in units where c = hbar = kBoltzmann = 1, the Planck temperature is
# TP = 1/sqrt(G) = 1.41678416E32 Kelvin
#    = sqrt(8pi) / sqrt(8piG) = sqrt(8pi) * M4
# that gives us 1 Kelvin measured in GeV
Kelvin = np.sqrt(Const_8Pi) * M4 / 1.41678416E32

# in units where c = hbar = 1, the Planck length is
# ellP = sqrt(G) = 1.61625518E-35 m
#      = sqrt(8piG) / sqrt(8pi) = 1 / (M4 * sqrt(8pi))
# that gives us 1 metre measured in 1/GeV
Metre = 1.0 / (M4 * np.sqrt(Const_8Pi) * 1.61625518E-35)
Kilometre = 1000.0 * Metre
Mpc = 3.08567758128E+19 * Kilometre

# in units where c = hbar = 1, the Planck mass is
# MP = 1/sqrt(G) = 2.17643424E-8 kg
#    = sqrt(8pi) / sqrt(8piG) = sqrt(8pi) * M4
# that gives us 1 kg measured in GeV
Kilogram = np.sqrt(Const_8Pi) * M4 / 2.17643424E-8
Gram = Kilogram / 1000.0
SolarMass = 1.98847E30 * Kilogram


# The *ModelParameters* class captures details of the 4D and 5D Planck masses, and uses these to compute derived
# quantities such as the brane tension and the crossover temperature (in GeV and Kelvin) from the quadratic
# Hubble regime to the linear regime. It also computes the AdS radius in inverse GeV and metres, assuming that
# the bulk cosmological constant is tuned to produce zero four-dimensional cosmological constant.
# This is important for deciding when the transition takes place from a "small" 5D black hole (approximated
# by Myers-Perry) to a "large", effectively 4D black hole (approximated by 4D Schwarzschild).
class ModelParameters:

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

        # M_ratio < 1 (usually mu << 1) is the M5/M4 ratio
        M_ratio = M5 / M4
        self.M_ratio = M_ratio

        # compute brane tension lambda = 6 mu^2 M5^4
        # note we can't actually call it lambda in Python, for which lambda is a reserved word
        self.tension = 6 * M_ratio*M_ratio * M5*M5*M5*M5

        # also compute the mass scale associated with the tension
        self.tension_scale = Const_4thRoot_6 * np.sqrt(M_ratio) * M5

        # compute crossover temperature from the quadratic to the linear regime, which occurs when rho = 2 lambda

        # we need the 4D radiation density constant (per degree of freedom) to convert temperature to energy density
        self.RadiationConstant = RadiationConstant4D
        self.StefanBoltzmannConstant4D = StefanBoltzmannConstant4D
        self.StefanBoltzmannConstant5D = StefanBoltzmannConstant4D

        # assume the crossover temperature is high enough that all SM particles are relativistic and in thermal
        # equilibrium, which should be good above Tcross = 200 GeV; we insist Tcross > 1E3 (see below),
        # which is hopefully enough headroom
        self.gstar = gstar_full_SM

        self.T_crossover = np.power(12.0 / (self.RadiationConstant * self.gstar), 1.0/4.0) * np.sqrt(M_ratio) * M5
        self.T_crossover_Kelvin = self.T_crossover / Kelvin

        # compute AdS radius ell and its inverse, mu
        self.mu = M_ratio * M_ratio * M5
        self.ell_AdS = 1.0 / self.mu
        self.ell_AdS_Metres = self.ell_AdS / Metre

        # compute mass-scale at which black holes cross over from 5D to 4D behaviour
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


# The *CosmologyEngine* class provides methods to compute the Hubble rate, Hubble length, horizon mass, etc.
class CosmologyEngine:

    def __init__(self, params: ModelParameters):
        self.params = params

    # compute the radiation energy density in GeV^4 from a temperature supplied in GeV
    # currently, we assume there are a fixed number of relativistic species
    def rho_radiation(self, T=None, log_T=None):
        # if T is not supplied, try to use log_T
        if T is not None:
            _T = T
        elif log_T is not None:
            _T = np.exp(log_T)
        else:
            raise RuntimeError('No temperature value supplied to CosmologyEngine.rho_radiation()')

        # check that supplied temperature is lower than the 5D Planck mass
        if T > self.params.M5:
            raise RuntimeError('Temperature T={temp:.3g} GeV is higher than the 5D Planck '
                               'mass M5={M5:.3g} GeV'.format(temp=T, M5=self.params.M5))

        return self.params.RadiationConstant * self.params.gstar * T * T * T * T

    # compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * np.sqrt(rho * (1.0 + rho / (2.0 * self.params.tension)))

    # compute the 4D-only Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble4(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * np.sqrt(rho)

    # compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    # the formula here is R_H = 1/H
    def R_Hubble(self, T=None, log_T=None):
        return 1.0 / self.Hubble(T, log_T)

    # compute the 4D-only Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    def R_Hubble4(self, T=None, log_T=None):
        return 1.0 / self.Hubble4(T, log_T)

    # compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
    # the formula here is M_H = (4/3) pi rho R_H^3, but we compute it directly to avoid multiple evaluations of rho
    def M_Hubble(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 \
              * np.power(1.0 + rho / (2.0 * self.params.tension), -3.0/2.0) / np.sqrt(rho)

        return M_H

    # compute the mass (in GeV) enclosed within the 4D-only Hubble length, at a time corresponding to a temperature supplied in GeV
    def M_Hubble4(self, T=None, log_T=None):
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 / np.sqrt(rho)

        return M_H


# class PBHModel represents the state of a PBH, which involves at least mass but possibly also charge and
# angular momentum. It can be used as an arithmetic type and and can be queried for other properties, such as
# the Hawking temperature.
#
# The crossover from 5D to 4D behaviour is taken to occur when the 5D Myers-Perry radius is equal to
# (4/3) * ell, where ell is the AdS curvature radius. This makes the black hole temperatures continuous.
class PBHModel:

    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    # capture (i) initial mass value, and (ii) a ModelParameters instance so we can decide whether we are in the 4D or
    # 5D regime based on the AdS radius.
    # The initial mass value can be specified in grams, kilograms, or GeV, but defaults to GeV
    def __init__(self, params: ModelParameters, mass: float, units='GeV'):
        self.params = params

        self._M4_over_M5_sqrt = np.sqrt(1.0/self.params.M_ratio)

        # assign current value
        self.set_value(mass, units)

        # check mass is larger than 4D Planck mass; there's no need to check the 5D Planck mass, because
        # we guarantee that M4 > M5
        if self.mass <= self.params.M4:
            raise RuntimeError('Initial black hole mass {mass} GeV should be larger than the 4D Planck mass '
                               '{MP} GeV in order that the PBH does not begin life as a '
                               'relic'.format(mass=self.mass, MP=self.params.M4))


    def set_value(self, mass: float, units='GeV'):
        if units not in self._mass_conversions:
            raise RuntimeError('PBHModel: unit "{unit}" not understood in constructor'.format(unit=units))

        units_to_GeV = self._mass_conversions[units]
        self.mass = mass * units_to_GeV


    # implement basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, PBHModel):
            return PBHModel(self.params, self.mass + other.mass)

        return PBHModel(self.params, self.mass + other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, PBHModel):
            return PBHModel(self.params, self.mass - other.mass)

        return PBHModel(self.params, self.mass - other)

    def __rsub__(self, other):
        if isinstance(other, PBHModel):
            return PBHModel(self.params, other.mass - self.mass)

        return PBHModel(self.params, other - self.mass)

    def __mul__(self, other):
        if isinstance(other, PBHModel):
            return NotImplemented

        return PBHModel(self.params, other * self.mass)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, PBHModel):
            return NotImplemented

        return PBHModel(self.params, self.mass / other)

    # query for the 5D Myers-Perry radius of the black hole, measured in 1/GeV
    @property
    def radius_5D(self):
        return Const_Radius_5D * np.sqrt(self.mass / self.params.M5) / self.params.M5

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

    # query for value of the dimensionless ratio Hrh in a 4D model
    def Hrh_4D(self, x, rho):
        return x / (1.0 + rho/(2.0*self.params.tension))

    # query for value of the dimensionless ratio Hrh in a 5D model
    def Hrh_5D(self, x, rho):
        return Const_Hrh_5D * np.sqrt(x) * self._M4_over_M5_sqrt * \
            np.power(rho / (1.0 + rho/(2.0*self.params.tension)), 1.0/4.0) / self.params.M5

    def Hrh(self, x, rho):
        if self.is_5D:
            return self.Hrh_4D(x, rho)

        else:
            return self.Hrh_5D(x, rho)


class StefanBoltzmannLifetimeModel:
    '''
    Evaluate RHS of mass evolution model using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: CosmologyEngine, accretion_efficiency_F=0.3,
                 use_effective_radius=True):
        '''
        Instantiate a StefanBoltzmannLifetimeModel object
        :param engine: a CosmologyEngine instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('StefanBoltzmannLifetimeModel: supplied CosmologyEngine instance is not usable')

        self.engine = engine
        self._params = engine.params
        self._tension = params.tension
        self._M4_4 = self._params.M4 * self._params.M4 * self._params.M4 * self._params.M4
        self._SB_4D = self._params.StefanBoltzmannConstant4D
        self._SB_5D = self._params.StefanBoltzmannConstant5D

        # create a PBHModel instamce; the value assigned to the mass doesn't matter
        self._M_PBH = PBHModel(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logx_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = np.exp(logT_rad)

        # also the PBH mass fraction
        x = np.exp(logx_asarray.item())

        # instantiate a PBHModel object to hold the PBH current state, and also the initial value of the
        # temperature
        M_Hubble = self.engine.M_Hubble(T=T_rad)
        self._M_PBH.set_value(x * M_Hubble, 'GeV')

        # compute radiation energy density
        rho = self.engine.rho_radiation(T=T_rad)

        # first term in evolution equation comes from variation of M_H
        dlogx_dlogT = 12.0 * (self._tension + rho) / (2.0 * self._tension + rho) - 4.0


        # ACCRETION

        Hrh = self._M_PBH.Hrh(x, rho)
        Hrh_sq = Hrh * Hrh

        alpha = self._M_PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        dlogx_dlogT -= (3.0/4.0) * self._accretion_efficiency_F * alpha_sq * Hrh_sq / x


        # EVAPORATION

        t = self._M_PBH.t
        t4 = t*t*t*t

        g4_evap = 2.0  # TODO: CURRENTLY APPROXIMATE - ASSUME ONLY RADIATES TO PHOTONS
        g5_evap = 5.0  # TODO: ASSUME ONLY RADIATES TO BULK GRAVITONS

        evap_prefactor = (alpha_sq/3.0) * rho * (1.0 + rho / (2.0 * self._tension)) / (x * t4 * Hrh_sq * self._M4_4)
        evap_dof = (g4_evap * self._SB_4D + Const_PiOver2 * alpha * g5_evap * self._SB_5D / t)

        dlogx_dlogT += evap_prefactor * evap_dof
        print('-- integrator called at x = {x:.5g}, M_PBH = {MPBH:.5g} gram, T = {T:.5g} GeV, returning dlogx_dlogT = {out:.5g}'.format(x=x, MPBH=self._M_PBH.mass/Gram, T=T_rad, out=dlogx_dlogT))

        return dlogx_dlogT


class LifetimeObserver:
    '''
    LifetimeObserver is a policy object that decides when to store data about the computed
    PBH model (i.e. mass as a function of T), and also checks whether the integration should abort
    because evaporation has proceeded to the point where a relic has formed
    '''
    # constructor captures CosmologyEngine instance. _sample_grid should be a numpy 1d array representing points where
    # we want to sample the solution M(T), and mass_grid is an (empty) numpy 1d array of the same shape
    # into which the answer will be written
    def __init__(self, engine: CosmologyEngine, sample_grid, mass_grid, x_grid):
        '''
        Instantiate a LifetimeObserver instance
        :param engine: CosmologyEngine instance to use for computations
        :param sample_grid: grid of sample points for independent variable (here log T)
        :param mass_grid: grid of sample points for dependent variable (here M)
        :param x_grid: grid of sample points for dependent variable (here x)
        '''
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('LifetimeObserver: supplied CosmologyEngine instance is not usable')

        if sample_grid.shape != mass_grid.shape:
            raise RuntimeError('LifetimeObserver: _sample_grid and mass_grid shapes do not match')

        # capture cosmology engine
        self._engine = engine

        # self.terminated is a flag that is set when the integration should terminate because a relic
        # has formed; self.relic_mass is the PBH mass where we declare a relic forms
        self.relic_mass = engine.params.M4
        self.terminated = False

        # capture reference to sample grid and data grid
        self._sample_grid = sample_grid
        self._mass_grid = mass_grid
        self._x_grid = x_grid

        self._sample_grid_length = sample_grid.size

        # self.sample_grid_current_index is an externally visible data member that exposes our current
        # position within the same grid
        self.sample_grid_current_index = 0

        # self.next_sample_point is an externally visible data member that exposes the value of the
        # next sample poiint
        if self._sample_grid_length > 0:
            self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
        else:
            self.next_sample_point = None

    # observation step should sample the solution if needed, and check whether the integration should end
    def __call__(self, logT_rad, logx_asarray):
        '''
        Execute an observation step. This should sample the solution if needed, storing the current value in
        self._mass_grid, and advance self.sample_grid_current_index (and update self.next_sample_point)
        :param logT_rad: current value of log T for the radiation bath
        :param logx_asarray: current value of log x, where x is the PBH mass fraction x = M/M_H
        :return:
        '''
        # for some calculations we cannot avoid using the temperature of the radiation bath
        # expressed in GeV
        T_rad = np.exp(logT_rad)

        # we also comtimes need the current value of x
        x = np.exp(logx_asarray.item())

        # compute current PBH mass
        M_PBH = x * self._engine.M_Hubble(T=T_rad)

        # write solution into M-grid if we have passed an observation point
        if self.next_sample_point is not None and logT_rad < self.next_sample_point:
            self._mass_grid[self.sample_grid_current_index] = M_PBH
            self._x_grid[self.sample_grid_current_index] = x

            self.sample_grid_current_index += 1
            if self.sample_grid_current_index < self._sample_grid_length:
                self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
            else:
                self.next_sample_point = None

        # check whether integration should halt because we have decreased the PBH mass below the 4D Planck scale M4.
        # If this happens, we either get a relic, or at least the standard calculation of Hawking radiation is
        # invalidated, so either way we should stop
        if M_PBH < self.relic_mass:
            self.terminated = True
            return -1

        return 0


class PBHLifetimeModel:

    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    # conversion factors into GeV for temperature units we understand
    _temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}


    def __init__(self, x_init, M_init, T_rad_init, LifetimeModel, num_samples=NumTSamplePoints):
        '''
        Capture initial valies
        :param x_init: initial value of mass fraction x = M/M_H
        :param M_init: initial PBH mass, expressed in GeV
        :param T_rad_init: temperature of radiation bath at formation, expressed in GeV
        :param LifetimeModel: model to use for lifetime calculations
        :param num_samples: number of samples to take
        '''
        # LifetimeModel should include an engine field to which we can refer
        self._engine = LifetimeModel.engine

        self.x_init = x_init
        self.M_init = M_init
        self.T_rad_init = T_rad_init

        # integration actually proceeds with log(x)
        self.log_x_init = np.log(x_init)

        # integration is done in terms of log(x) and log(T), where x = M/M_H(T) is the PBH mass expressed
        # as a fraction of the Hubble mass M_H
        self.logT_rad_init = np.log(T_rad_init)

        # sample grid runs from initial temperature of the radiation bath at formation,
        # down to current CMB temmperature T_CMB
        self.T_min = T_CMB * Kelvin
        self.logT_min = np.log(self.T_min)

        self.T_sample_points = np.geomspace(T_rad_init, self.T_min, num_samples)
        self.logT_sample_points = np.log(self.T_sample_points)

        # reserve space for mass history
        self.M_sample_points = np.zeros_like(self.logT_sample_points)
        self.x_sample_points = np.zeros_like(self.logT_sample_points)

        # prepare an observer object using these sample points
        _observer = LifetimeObserver(self._engine, self.logT_sample_points, self.M_sample_points, self.x_sample_points)

        # run the integration
        self._integrate(LifetimeModel, _observer)


    def _integrate(self, LifetimeModel, Observer):
        '''

        :param LifetimeModel: callable representing RHS of ODE system
        :param Observer: callable representing solution observer (to record solution at specified sample points)
        :return:
        '''
        # set up stepper; need to use on that supports solout, which the SUNDIALS ones don't seem to do
        stepper = ode(LifetimeModel).set_integrator('dopri5', rtol=1E-10, nsteps=5000)
        stepper.set_solout(Observer)

        # set up initial conditions for the PBH and the radiation bath
        # to keep the numerics sensible, we can't run the integration directly in grams; the numbers get too large,
        # making the integrator need a very small stepsize to keep up
        stepper.set_initial_value(self.log_x_init, self.logT_rad_init)

        # integrate down to the present CMB temperature, or when the observer notices that the PBH
        # mass has decreased below M4

        while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
                and not Observer.terminated:
            stepper.integrate(Observer.next_sample_point - 0.001)

        # if there was an integration failure, raise an exception
        if not stepper.successful():
            raise RuntimeError('PBH lifetime calculation failed due to an integration error, '
                               'code = {code}'.format(code=stepper.get_return_code()))

        # truncate unused sample points at end of x_sample_points
        index = Observer.sample_grid_current_index
        if index < self.T_sample_points.size:
            np.resize(self.T_sample_points, index)
            np.resize(self.logT_sample_points, index)
            np.resize(self.M_sample_points, index)
            np.resize(self.x_sample_points, index)

        # if the observer terminated the integration, this is because the PBH evaporation proceeded
        # to the point where we produce a relic
        if Observer.terminated:
            self.T_lifetime = np.exp(stepper.t)
            return

        self.T_lifetime = None


    # produce plot of PBH mass over its lifetime, as a function of temperature T
    def mass_plot(self, filename, mass_units='gram', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        T_values = self.T_sample_points / temperature_units_to_GeV
        # M_values = self.M_sample_points / mass_units_to_GeV
        M_values = self.x_sample_points

        plt.figure()
        plt.loglog(T_values, M_values,
                   label='initial mass = {i:.3g} {unit}'.format(i=self.M_init / mass_units_to_GeV,
                                                                unit=mass_units))
        plt.xlabel('Temperature T / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH mass / {unit}'.format(unit=mass_units))
        plt.legend()
        plt.savefig(filename)


# class PBHInstance captures details of a PBH that forms at a specified initial temperature
# (which we can map to an initial mass and a lengthscale)
class PBHInstance:
    # capture cosmology engine instance and formation temperature of the PBH, measured in GeV
    # T_rad_init: temperature of radiation bath at PBH formation
    # accretion_efficiency: accretion efficiency factor F in Bondi-Hoyle-Lyttleton model
    # collapse_fraction: fraction of Hubble volume that collapses to PBH
    # num_sample_ponts: number of T samples to take
    def __init__(self, engine: CosmologyEngine, T_rad_init: float, accretion_efficiency_F=0.5,
                 collapse_fraction_f=0.5, delta=0.0, num_samples=NumTSamplePoints):
        self.engine = engine
        self.accretion_efficiency_F = accretion_efficiency_F

        _x_init = collapse_fraction_f * (1.0 + delta)
        _M_Hubble = engine.M_Hubble(T=T_rad_init)
        _M_init = _x_init * _M_Hubble

        self.lifetimes \
            = {'standard': PBHLifetimeModel(_x_init, _M_init, T_rad_init,
                                            StefanBoltzmannLifetimeModel(self.engine,
                                                                         accretion_efficiency_F=accretion_efficiency_F,
                                                                         use_effective_radius=True),
                                            num_samples=num_samples)}


# generate a plot of PBH formation mass vs. formation temperature
def PBHMassPlot(M5, Tlo=1E3, Thi=None, units='gram', collapse_fraction_f=0.5):
    # check desired units are sensible
    if units not in ['gram', 'kilogram', 'GeV']:
        raise RuntimeError('PBHMassPlot: unit "{unit}" not understood in constructor'.format(unit=units))

    if Thi is None:
        Thi = M5

    # build a dictionary of unit conversion coefficients
    units_conversion = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    params = ModelParameters(M5)
    engine = CosmologyEngine(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    unit = units_conversion[units]

    M_values = [collapse_fraction_f * engine.M_Hubble(T=T) / unit for T in T_range]
    M4_values = [collapse_fraction_f * engine.M_Hubble4(T=T) / unit for T in T_range]

    plt.figure()
    plt.loglog(T_range, M_values, label='Randall-Sundrum')
    plt.loglog(T_range, M4_values, label='Standard 4D')
    plt.xlabel('Temperature T / GeV')
    plt.ylabel('PBH mass at formation / {units}'.format(units=units))
    plt.legend()
    plt.savefig('formation-mass.pdf')


# generate a plot of PBH formation lengthscale vs. formation temperature
def PBHLengthscalePlot(M5, Tlo=4E3, Thi=None, units='kilometre'):
    # check desired units are sensible
    if units not in ['metre', 'kilometre', 'Mpc']:
        raise RuntimeError('PBHLengthscalePlot: unit "{unit}" not understood in constructor'.format(unit=units))

    if Thi is None:
        Thi = M5

    # build a dictionary of unit conversion coefficients
    units_conversion = {'metre': Metre, 'kilometre': Kilometre, 'Mpc': Mpc}

    params = ModelParameters(M5)
    engine = CosmologyEngine(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    unit = units_conversion[units]

    R_values = [engine.R_Hubble(T=T) / unit for T in T_range]
    R4_values = [engine.R_Hubble4(T=T) / unit for T in T_range]

    plt.figure()
    plt.loglog(T_range, R_values, label='Randall-Sundrum')
    plt.loglog(T_range, R4_values, label='Standard 4D')
    plt.xlabel('Temperature T / GeV')
    plt.ylabel('Collapse lengthscale / {units}'.format(units=units))
    plt.legend()
    plt.savefig('formation-lengthscale.pdf')


# generate a plot of PBH formation mass vs. formation lengthscale
def PBHMassScaleRelation(M5, Tlo=4E3, Thi=None, length_units='kilometre', mass_units='gram', collapse_fraction_f=0.5):
    # check desired units are sensible
    if length_units not in ['metre', 'kilometre', 'Mpc']:
        raise RuntimeError('PBHLengthscalePlot: unit "{unit}" not understood in constructor'.format(unit=length_units))

    if mass_units not in ['gram', 'kilogram', 'GeV']:
        raise RuntimeError('PBHMassPlot: unit "{unit}" not understood in constructor'.format(unit=mass_units))

    if Thi is None:
        Thi = M5

    # build a dictionary of unit conversion coefficients
    length_conversion = {'metre': Metre, 'kilometre': Kilometre, 'Mpc': Mpc}
    mass_conversion = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    params = ModelParameters(M5)
    engine = CosmologyEngine(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    length_unit = length_conversion[length_units]
    mass_unit = mass_conversion[mass_units]

    R_values = [engine.R_Hubble(T=T) / length_unit for T in reversed(T_range)]
    R4_values = [engine.R_Hubble4(T=T) / length_unit for T in reversed(T_range)]

    M_values = [collapse_fraction_f * engine.M_Hubble(T=T) / mass_unit for T in reversed(T_range)]
    M4_values = [collapse_fraction_f * engine.M_Hubble4(T=T) / mass_unit for T in reversed(T_range)]

    plt.figure()
    plt.loglog(R_values, M_values, label='Randall-Sundrum')
    plt.loglog(R4_values, M4_values, label='Standard 4D')
    plt.xlabel('Lengthscale $\ell$ / {units}'.format(units=length_units))
    plt.ylabel('PBH mass at formation / {units}'.format(units=mass_units))
    plt.legend()
    plt.savefig('PBH-mass-lengthscale.pdf')


# Generate plots
PBHMassPlot(5E12)
PBHLengthscalePlot(5E12)
PBHMassScaleRelation(5E12)

# compute PBH evolution
params = ModelParameters(1E14)
print(params)
engine = CosmologyEngine(params)

sample = PBHInstance(engine, 1E10, accretion_efficiency_F=0.1, collapse_fraction_f=0.5)
standard = sample.lifetimes['standard']
standard.mass_plot('PBH_T1E10_mass.pdf')
