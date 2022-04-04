import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt

import time


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


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


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


class StefanBoltzmann5DLifetimeModel:
    '''
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum cosmology),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: CosmologyEngine, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True):
        '''
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a CosmologyEngine instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('StefanBoltzmann5DLifetimeModel: supplied CosmologyEngine instance is not usable')

        self.engine = engine
        self._params = engine.params
        self._SB_4D = self._params.StefanBoltzmannConstant4D
        self._SB_5D = self._params.StefanBoltzmannConstant5D

        # create a PBHModel instamce; the value assigned to the mass doesn't matter
        self._M_PBH = PBHModel(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        self._logM_end = np.log(self._params.M4)

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = np.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._M_PBH to its value
        logM = logM_asarray.item()
        M_PBH = np.exp(logM)
        self._M_PBH.set_value(M_PBH, 'GeV')

        # compute horizon radius in 1/GeV
        rh = self._M_PBH.radius
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = self._M_PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        # ACCRETION

        dlogM_dlogT = -np.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho / (self._M_PBH.mass * H)


        # EVAPORATION

        t = self._M_PBH.t
        t4 = t*t*t*t

        g4_evap = 2.0  # TODO: CURRENTLY APPROXIMATE - ASSUME ONLY RADIATES TO PHOTONS
        g5_evap = 5.0  # TODO: ASSUME ONLY RADIATES TO BULK GRAVITONS

        evap_prefactor = Const_4Pi * alpha_sq / (self._M_PBH.mass * H * t4 * rh_sq)
        evap_dof = (g4_evap * self._SB_4D + Const_PiOver2 * alpha * g5_evap * self._SB_5D / t)

        dlogM_dlogT += evap_prefactor * evap_dof / (2.6 if self._use_Page_suppression else 1.0)

        # x = self._M_PBH.mass / self.engine.M_Hubble(T=T_rad)
        # evap_to_accrete = 4.0 / (self._accretion_efficiency_F * t4 * rh_sq * rh_sq * rho)
        #
        # T_Hawking = self._M_PBH.T_Hawking
        #
        # print('-- integrator called at x = {x:.5g}, M_PBH = {MPBHGeV:.5g} GeV = {MPBHgram:.5g} gram, '
        #       'T = {TGeV:.5g} GeV = {TKelvin:.5g} Kelvin, returning dlogM_dlogT = {out:.5g}, dM/dT = {nolog:.5g}, '
        #       'evap/accrete = {ratio:.5g}, T_Hawking = {THawkGeV:.5g} = {THawkKelvin:.5g} '
        #       'Kelvin'.format(x=x, MPBHGeV=self._M_PBH.mass, MPBHgram=self._M_PBH.mass / Gram,
        #                       TGeV=T_rad, TKelvin=T_rad / Kelvin, out=dlogM_dlogT,
        #                       nolog=self._M_PBH.mass * dlogM_dlogT / T_rad, ratio=evap_to_accrete,
        #                       THawkGeV=T_Hawking, THawkKelvin=T_Hawking/Kelvin))

        return dlogM_dlogT


class StefanBoltzmann4DLifetimeModel:
    '''
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional cosmology),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: CosmologyEngine, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True):
        '''
        Instantiate a StefanBoltzmann4DLifetimeModel object
        :param engine: a CosmologyEngine instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('StefanBoltzmann4DLifetimeModel: supplied CosmologyEngine instance is not usable')

        self.engine = engine
        self._params = engine.params
        self._SB_4D = self._params.StefanBoltzmannConstant4D

        # create a PBHModel instamce; the value assigned to the mass doesn't matter
        self._M_PBH = PBHModel(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = True

        self._logM_end = np.log(self._params.M4)

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = np.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._M_PBH to its value
        logM = logM_asarray.item()
        M_PBH = np.exp(logM)
        self._M_PBH.set_value(M_PBH, 'GeV')

        # compute horizon radius in 1/GeV
        rh = self._M_PBH.radius_4D
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble4(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = Const_Reff_4D if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        # ACCRETION

        dlogM_dlogT = -np.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho / (self._M_PBH.mass * H)


        # EVAPORATION

        t = Const_4Pi   # only need 4D result
        t4 = t*t*t*t

        g4_evap = 2.0  # TODO: CURRENTLY APPROXIMATE - ASSUME ONLY RADIATES TO PHOTONS

        evap_prefactor = Const_4Pi * alpha_sq / (self._M_PBH.mass * H * t4 * rh_sq)
        evap_dof = g4_evap * self._SB_4D

        dlogM_dlogT += evap_prefactor * evap_dof / (2.6 if self._use_Page_suppression else 1.0)

        # x = self._M_PBH.mass / self.engine.M_Hubble4(T=T_rad)
        # evap_to_accrete = 4.0 / (self._accretion_efficiency_F * t4 * rh_sq * rh_sq * rho)
        #
        # T_Hawking = self._M_PBH.T_Hawking
        #
        # print('-- integrator called at x = {x:.5g}, M_PBH = {MPBHGeV:.5g} GeV = {MPBHgram:.5g} gram, '
        #       'T = {TGeV:.5g} GeV = {TKelvin:.5g} Kelvin, returning dlogM_dlogT = {out:.5g}, dM/dT = {nolog:.5g}, '
        #       'evap/accrete = {ratio:.5g}, T_Hawking = {THawkGeV:.5g} = {THawkKelvin:.5g} '
        #       'Kelvin'.format(x=x, MPBHGeV=self._M_PBH.mass, MPBHgram=self._M_PBH.mass / Gram,
        #                       TGeV=T_rad, TKelvin=T_rad / Kelvin, out=dlogM_dlogT,
        #                       nolog=self._M_PBH.mass * dlogM_dlogT / T_rad, ratio=evap_to_accrete,
        #                       THawkGeV=T_Hawking, THawkKelvin=T_Hawking/Kelvin))

        return dlogM_dlogT


class LifetimeObserver:
    '''
    LifetimeObserver is a policy object that decides when to store data about the computed
    PBH model (i.e. mass as a function of T), and also checks whether the integration should abort
    because evaporation has proceeded to the point where a relic has formed
    '''
    # constructor captures CosmologyEngine instance. _sample_grid should be a numpy 1d array representing points where
    # we want to sample the solution M(T), and mass_grid is an (empty) numpy 1d array of the same shape
    # into which the answer will be written
    def __init__(self, engine: CosmologyEngine, sample_grid, mass_grid, x_grid, relic_mass, use_4D=False):
        '''
        Instantiate a LifetimeObserver instance
        :param engine: CosmologyEngine instance to use for computations
        :param sample_grid: soln_grid of sample points for independent variable (here log T)
        :param mass_grid: soln_grid of sample points for dependent variable (here M)
        :param x_grid: soln_grid of sample points for dependent variable (here x)
        :param use_4D: set to true to use 4D Hubble mass in calculation of mass fraction x
        '''
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('LifetimeObserver: supplied CosmologyEngine instance is not usable')

        if sample_grid.shape != mass_grid.shape:
            raise RuntimeError('LifetimeObserver: _sample_grid and mass_grid shapes do not match')

        # capture cosmology engine
        self._engine = engine

        # capture use_4D setting
        self._use_4D = use_4D

        # self.terminated is a flag that is set when the integration should terminate because a relic
        # has formed; self.relic_mass records the PBH mass where we declare a relic forms
        self.relic_mass = relic_mass
        self.terminated = False

        # capture reference to sample soln_grid and data soln_grid
        self._sample_grid = sample_grid
        self._mass_grid = mass_grid
        self._x_grid = x_grid

        self._sample_grid_length = sample_grid.size

        # self.sample_grid_current_index is an externally visible data member that exposes our current
        # position within the same soln_grid
        self.sample_grid_current_index = 0

        # self.next_sample_point is an externally visible data member that exposes the value of the
        # next sample poiint
        if self._sample_grid_length > 0:
            self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
        else:
            self.next_sample_point = None

    # observation step should sample the solution if needed, and check whether the integration should end
    def __call__(self, logT_rad, logM_asarray):
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

        # extract current value of PBH mass, in GeV
        M_PBH = np.exp(logM_asarray.item())

        # write solution into M-soln_grid if we have passed an observation point
        if self.next_sample_point is not None and logT_rad < self.next_sample_point:
            # compute mass as a fraction of the Hubble volume mass
            x = M_PBH / (self._engine.M_Hubble4(T=T_rad) if self._use_4D else self._engine.M_Hubble(T=T_rad))

            # store these values
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


def Solve_4D_T(Ti, Mi, Mf, gstar, a, tension, g4, sigma4, g5, sigma5, M4, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = np.sqrt(a*gstar)

    Ti_sq = Ti*Ti
    Ti_4 = Ti_sq*Ti_sq

    tension_sqrt = np.sqrt(tension)

    alpha_sq = alpha*alpha

    M4_sq = M4*M4

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * np.power(1.0 - Mf_over_Mi*Mf_over_Mi*Mf_over_Mi, 1.0/3.0)
    DeltaM_over_M4 = DeltaM / M4
    DeltaM_over_M4_3 = DeltaM_over_M4 * DeltaM_over_M4 * DeltaM_over_M4

    g_factor = 8.0*g4*sigma4 + g5*alpha*sigma5

    A_const = 64.0*np.sqrt(2.0/3.0)*np.pi/3.0

    A1 = np.sqrt(a_gstar*Ti_4 + 2.0*tension)/Ti_sq
    A2 = A_const * a_gstar_sqrt * tension_sqrt * DeltaM_over_M4_3 / (M4_sq * alpha_sq * g_factor)
    A = A1 + A2

    A_sq = A*A

    return np.power(2.0*tension / (A_sq - a_gstar), 1.0/4.0)

def Solve_5D_T(Ti, Mi, Mf, gstar, a, tension, g4, sigma4, g5, sigma5, M4, M5, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = np.sqrt(a * gstar)

    Ti_sq = Ti*Ti
    Ti_4 = Ti_sq*Ti_sq

    tension_sqrt = np.sqrt(tension)

    alpha_sq = alpha*alpha

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * np.sqrt(1.0 - Mf_over_Mi*Mf_over_Mi)
    DeltaM_over_M5 = DeltaM / M5
    DeltaM_over_M5_sq = DeltaM_over_M5*DeltaM_over_M5

    g_factor = 4.0*g4*sigma4 + g5*alpha*sigma5

    A_const = 16.0*np.sqrt(2.0/3.0)*np.pi/3.0

    A1 = np.sqrt(a_gstar*Ti_4 + 2.0*tension) / Ti_sq
    A2 = A_const * a_gstar_sqrt * tension_sqrt * DeltaM_over_M5_sq / (M4 * M5 * alpha_sq * g_factor)
    A = A1 + A2

    A_sq = A*A

    return np.power(2.0*tension / (A_sq - a_gstar), 1.0/4.0)


class PBHLifetimeModel:
    def __init__(self, M_init, T_rad_init, LifetimeModel, num_samples=NumTSamplePoints, use_4D=False):
        '''
        Capture initial values
        :param M_init: initial PBH mass, expressed in GeV
        :param T_rad_init: temperature of radiation bath at formation, expressed in GeV
        :param LifetimeModel: model to use for lifetime calculations
        :param num_samples: number of samples to extract
        :param use_4D: should be True to use 4D Hubble mass in observer calculation of mass fraction x
        '''
        # LifetimeModel should include an engine field to which we can refer
        self._engine = LifetimeModel.engine
        self._params = self._engine.params
        self._use_4D = use_4D

        self.M_init = M_init
        self.T_rad_init = T_rad_init

        # integration actually proceeds with log(x)
        self.logM_init = np.log(M_init)

        # integration is done in terms of log(x) and log(T), where x = M/M_H(T) is the PBH mass expressed
        # as a fraction of the Hubble mass M_H
        self.logT_rad_init = np.log(T_rad_init)

        # sample soln_grid runs from initial temperature of the radiation bath at formation,
        # down to current CMB temmperature T_CMB
        self.T_min = T_CMB * Kelvin
        self.logT_min = np.log(self.T_min)

        self.T_sample_points = np.geomspace(T_rad_init, self.T_min, num_samples)
        self.logT_sample_points = np.log(self.T_sample_points)

        # reserve space for mass history, expressed as a PBH mass in GeV and as a fraction x of the
        # currently Hubble mass M_H, x = M/M_H
        self.M_sample_points = np.zeros_like(self.logT_sample_points)
        self.x_sample_points = np.zeros_like(self.logT_sample_points)

        # set lifetime to default value of None, indicating that we could not compute it; we'll overwrite
        # this value later
        self.T_lifetime = None

        # if we have to use an analytic solution to get all the way down to the relic scale,
        # keep track of how much we needed to shift by
        self.T_shift = None

        # set compute time to None; will be overwritten later
        self.compute_time = None

        # prepare an observer object using these sample points, using a relic scale set at the
        self._relic_scale = self._params.M4
        observer = LifetimeObserver(self._engine, self.logT_sample_points, self.M_sample_points, self.x_sample_points,
                                    self._relic_scale, use_4D=use_4D)

        # run the integration
        self._integrate(LifetimeModel, observer, use_4D)


    def _integrate(self, LifetimeModel, Observer, use_4D=False):
        '''

        :param LifetimeModel: callable representing RHS of ODE system
        :param Observer: callable representing solution observer (to record solution at specified sample points)
        :param use_4D: assume 4D evolution
        :return:
        '''
        # set up stepper; need to use on that supports solout, which the SUNDIALS ones don't seem to do
        stepper = ode(LifetimeModel).set_integrator('dopri5', rtol=1E-8, nsteps=5000)
        stepper.set_solout(Observer)

        # set up initial conditions for the PBH and the radiation bath
        # to keep the numerics sensible, we can't run the integration directly in grams; the numbers get too large,
        # making the integrator need a very small stepsize to keep up
        stepper.set_initial_value(self.logM_init, self.logT_rad_init)

        with Timer() as timer:
            # integrate down to the present CMB temperature, or when the observer notices that the PBH
            # mass has decreased below M4

            # with np.errstate(over='raise', divide='raise'):
            #     try:
            #         while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
            #                 and not Observer.terminated:
            #             stepper.integrate(Observer.next_sample_point - 0.001)
            #     except FloatingPointError as e:
            #         print('Floating point error: {msg}'.format(msg=str(e)))
            #         print('  -- at Minit = {Minit}, T_rad = {Tinit}, M5={M5}'.format(Minit=self.M_init_5D, Tinit=self.T_rad_init, M5=LifetimeModel._params.M5))
            #
            #         # leave lifetime as null to indicate that numerical results were unreliable here
            #         return

            while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
                    and not Observer.terminated:
                stepper.integrate(Observer.next_sample_point - 0.001)

        self.compute_time = timer.interval

        # truncate unused sample points at end of x_sample_points
        index = Observer.sample_grid_current_index
        if index < self.T_sample_points.size:
            np.resize(self.T_sample_points, index)
            np.resize(self.logT_sample_points, index)
            np.resize(self.M_sample_points, index)
            np.resize(self.x_sample_points, index)

        # if the observer terminated the integration, this is because the PBH evaporation proceeded
        # to the point where we produce a relic, so we can record the lifetime and exit
        if Observer.terminated:
            self.T_lifetime = np.exp(stepper.t)
            return

        if stepper.successful():
            raise RuntimeError('Observer did not terminate, but integration ended without failure code')

        # if there was an integration failure, this is possibly because of a genuine problem, or possibly
        # because we could not resolve the final stages of the integration - because that is very close
        # to a singularity of the ODE system
        code = stepper.get_return_code()
        if code != -3:
            raise RuntimeError('PBH lifetime calculation failed due to an integration error at '
                               'T = {T:.5g} GeV = {TK:.5g} Kelvin, '
                               'code = {code}'.format(T=np.exp(stepper.t), TK=np.exp(stepper.t)/Kelvin,
                                                      code=stepper.get_return_code()))

        # this code corresponds to "step size becomes too small", which we interpret to mean
        # that we're close to the point of evaporation down to a relic

        params: ModelParameters = self._params

        # get current PBH mass in GeV
        M = np.exp(stepper.y.item())
        M_PBH = PBHModel(self._engine.params, M, 'GeV')

        # get current radiation temperature in GeV
        Ti = np.exp(stepper.t)

        # if the evolution is 4D only, don't use 5D step
        if self._use_4D:
            # evaporation is four-dimensional all the way down to the relic scale
            T_final = Solve_4D_T(Ti, M, self._relic_scale, params.gstar, params.RadiationConstant,
                                 params.tension, 2.0, params.StefanBoltzmannConstant4D,
                                 5.0, params.StefanBoltzmannConstant5D, params.M4,
                                 Const_Reff_4D if LifetimeModel._use_effective_radius else 1.0)

            DeltaT = Ti - T_final

        else:
            # if the PBH is already in the 5D regime, it stays in that regime all the way down to a relic
            if M_PBH.is_5D:
                # assume evaporation is five-dimensional all the way down to the relic scale
                T_final = Solve_5D_T(Ti, M, self._relic_scale, params.gstar, params.RadiationConstant,
                                     params.tension, 2.0, params.StefanBoltzmannConstant4D,
                                     5.0, params.StefanBoltzmannConstant5D, params.M4, params.M5,
                                     Const_Reff_5D if LifetimeModel._use_effective_radius else 1.0)

                DeltaT = Ti - T_final

            # otherwise there is a period of 4D evolution, followed by a period of 5D evolution
            else:
                # evolution is 4D down to the transition scale, then 5D down to the relic scale.
                # The relic scale is always above the transition scale, assuming that is taken to be
                # the 4D mass scale
                T_transition = Solve_4D_T(Ti, M, params.M_transition, params.gstar, params.RadiationConstant,
                                          params.tension, 2.0, params.StefanBoltzmannConstant4D,
                                          5.0, params.StefanBoltzmannConstant5D, params.M4,
                                          Const_Reff_4D if LifetimeModel._use_effective_radius else 1.0)

                T_final = Solve_5D_T(T_transition, params.M_transition, self._relic_scale, params.gstar,
                                     params.RadiationConstant, params.tension, 2.0, params.StefanBoltzmannConstant4D,
                                     5.0, params.StefanBoltzmannConstant5D, params.M4, params.M5,
                                     Const_Reff_5D if LifetimeModel._use_effective_radius else 1.0)

                DeltaT = Ti - T_final

        np.append(self.T_sample_points, T_final)
        np.append(self.logT_sample_points, np.log(T_final))
        np.append(self.M_sample_points, self._relic_scale)
        np.append(self.x_sample_points, self._relic_scale / self._engine.M_Hubble(T=T_final))

        self.T_lifetime = T_final
        self.T_shift = DeltaT


# class PBHInstance captures details of a PBH that forms at a specified initial temperature
# (which we can map to an initial mass and a lengthscale)
class PBHInstance:
    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    # conversion factors into GeV for temperature units we understand
    _temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}


    # capture cosmology engine instance and formation temperature of the PBH, measured in GeV
    # T_rad_init: temperature of radiation bath at PBH formation
    # accretion_efficiency: accretion efficiency factor F in Bondi-Hoyle-Lyttleton model
    # collapse_fraction: fraction of Hubble volume that collapses to PBH
    # num_sample_ponts: number of T samples to take
    def __init__(self, engine: CosmologyEngine, T_rad_init: float, accretion_efficiency_F=0.5,
                 collapse_fraction_f=0.5, delta=0.0, num_samples=NumTSamplePoints):
        self._engine = engine
        self._params = engine.params
        self.accretion_efficiency_F = accretion_efficiency_F

        # x = f (1+delta) is the fraction of the Hubble volume that initially collapses to form the PBH
        x_init = collapse_fraction_f * (1.0 + delta)

        # get mass of Hubble volume expressed in GeV
        M_Hubble = engine.M_Hubble(T=T_rad_init)
        M_Hubble4 = engine.M_Hubble4(T=T_rad_init)

        # compute initial mass in GeV
        M_init_5D = x_init * M_Hubble
        self.M_init_5D = M_init_5D

        M_init_4D = x_init * M_Hubble4
        self.M_init_4D = M_init_4D

        # set up different lifetime models - initially we are only using a Stefan-Boltzmann version
        sb_5D = StefanBoltzmann5DLifetimeModel(self._engine, accretion_efficiency_F=accretion_efficiency_F,
                                               use_effective_radius=True, use_Page_suppression=True)
        sb_4D = StefanBoltzmann4DLifetimeModel(self._engine, accretion_efficiency_F=accretion_efficiency_F,
                                               use_effective_radius=True, use_Page_suppression=True)

        self.lifetimes = {'StefanBoltzmann5D': PBHLifetimeModel(M_init_5D, T_rad_init, sb_5D, num_samples=num_samples),
                          'StefanBoltzmann4D': PBHLifetimeModel(M_init_4D, T_rad_init, sb_4D, num_samples=num_samples, use_4D=True)}


    # produce plot of PBH mass over its lifetime, as a function of temperature T
    def mass_plot(self, filename, histories=None, mass_units='gram', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        # if no histories specifies, plot them all
        if histories is None:
            histories = self.lifetimes.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        for label in histories:
            if label in self.lifetimes:
                history = self.lifetimes[label]
                T_values = history.T_sample_points / temperature_units_to_GeV
                M_values = history.M_sample_points / mass_units_to_GeV

                plt.loglog(T_values, M_values, label='{key}'.format(key=label))

        plt.xlabel('Temperature T / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH mass / {unit}'.format(unit=mass_units))
        plt.legend()
        plt.savefig(filename)
