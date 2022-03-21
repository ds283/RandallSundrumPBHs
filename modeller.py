import numpy as np
from scipy.integrate import ode

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial

sns.set()

# introduce fixed constants
Const_8Pi = 8.0 * np.pi
Const_4Pi = 4.0 * np.pi
Const_2Pi = 2.0 * np.pi
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
    def rho_radiation(self, T):
        # check that supplied temperature is lower than the 5D Planck mass
        if T > self.params.M5:
            raise RuntimeError('Temperature T={temp:.3g} GeV is higher than the 5D Planck '
                               'mass M5={M5:.3g} GeV'.format(temp=T, M5=self.params.M5))

        return self.params.RadiationConstant * self.params.gstar * T * T * T * T

    # compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble(self, T):
        rho = self.rho_radiation(T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * np.sqrt(rho * (1.0 + rho / (2.0 * self.params.tension)))

    # compute the 4D-only Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
    def Hubble4(self, T):
        rho = self.rho_radiation(T)

        return 1.0 / (Const_Sqrt_3 * self.params.M4) * np.sqrt(rho)

    # compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    # the formula here is R_H = 1/H
    def R_Hubble(self, T):
        return 1.0 / self.Hubble(T)

    # compute the 4D-only Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
    def R_Hubble4(self, T):
        return 1.0 / self.Hubble4(T)

    # compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
    # the formula here is M_H = (4/3) pi rho R_H^3, but we compute it directly to avoid multiple evaluations of rho
    def M_Hubble(self, T):
        rho = self.rho_radiation(T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 \
              * np.power(1.0 + rho / (2.0 * self.params.tension), -3.0/2.0) / np.sqrt(rho)

        return M_H

    # compute the mass (in GeV) enclosed within the 4D-only Hubble length, at a time corresponding to a temperature supplied in GeV
    def M_Hubble4(self, T):
        rho = self.rho_radiation(T)
        M_H = Const_M_H * self.params.M4 * self.params.M4 * self.params.M4 / np.sqrt(rho)

        return M_H


# class PBHState represents the state of a PBH, which involves at least mass but possibly also charge and
# angular momentum. It can be used as an arithmetic type and and can be queried for other properties, such as
# the Hawking temperature.
#
# The crossover from 5D to 4D behaviour is taken to occur when the 5D Myers-Perry radius is equal to
# (4/3) * ell, where ell is the AdS curvature radius. This makes the black hole temperatures continuous.
class PBHState:
    # capture (i) initial mass value, and (ii) a ModelParameters instance so we can decide whether we are in the 4D or
    # 5D regime based on the AdS radius.
    # The initial mass value can be specified in grams, kilograms, or GeV, but defaults to GeV
    def __init__(self, params: ModelParameters, mass: float, units='GeV'):
        if units not in ['gram', 'kilogram', 'GeV']:
            raise RuntimeError('PBHState: unit "{unit}" not understood in constructor'.format(unit=units))

        unit_conversion = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}
        unit = unit_conversion[units]

        self.params = params
        self.mass = mass * unit

        # check mass is larger than 4D Planck mass; there's no need to check the 5D Planck mass, because
        # we guarantee that M4 > M5
        if self.mass <= self.params.M4:
            raise RuntimeError('Initial black hole mass {mass} GeV should be larger than the 4D Planck mass '
                               '{MP} GeV in order that the PBH does not begin life as a '
                               'relic'.format(mass=self.mass, MP=self.params.M4))

    # implement basic arithmetic operations
    def __add__(self, other):
        if isinstance(other, PBHState):
            return PBHState(self.params, self.mass + other.mass)

        return PBHState(self.params, self.mass + other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, PBHState):
            return PBHState(self.params, self.mass - other.mass)

        return PBHState(self.params, self.mass - other)

    def __rsub__(self, other):
        if isinstance(other, PBHState):
            return PBHState(self.params, other.mass - self.mass)

        return PBHState(self.params, other - self.mass)

    def __mul__(self, other):
        if isinstance(other, PBHState):
            return NotImplemented

        return PBHState(self.params, other * self.mass)

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, PBHState):
            return NotImplemented

        return PBHState(self.params, self.mass / other)

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

class PBH_lifetime_RHS:
    # constructor captures CosmologyEngine instance and sets flags for the integration:
    #   accretion_efficiency_F: efficiency factor for accretion
    #   use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
    #   use_greybody_factors: whether Hawking radiation should be computed using greybody factors, or just the
    #                         Stefan-Boltzmann law
    def __init__(self, engine: CosmologyEngine, accretion_efficiency_F=1.0,
                 use_effective_radius=True, use_greybody_factors=True):
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('PBH_lifetime_RHS: supplied CosmologyEngine instance is not usable')

        self.engine = engine

        self.accretion_efficiency_F = accretion_efficiency_F

        self.use_effective_radius = use_effective_radius
        self.use_greybody_factors = use_greybody_factors

    # step the PBH mass, accounting for accretion and evaporation
    def eval(self, T_rad, M_PBH_asarray):
        # instantiate a PBHState object to hold the PBH current state, and also the initial value of the
        # temperature
        M_PBH = PBHState(self.engine.params, M_PBH_asarray.item(), 'GeV')

        print('-- integrator being called at T_rad = {TradGeV:.5g} GeV = {TradKelvin:.5g} Kelvin, '
              'mass = {MassGeV:.5g} = {MassGrams:.5g} grams'.format(TradGeV=T_rad, TradKelvin=T_rad/Kelvin,
                                                              MassGeV=M_PBH.mass, MassGrams=M_PBH.mass/Gram))

        # compute Hubble rate at this radiation temperature
        Hubble = self.engine.Hubble(T_rad)

        # compute black hole radius (accounting for 5D -> 4D transition)
        R_H = M_PBH.radius
        reff = M_PBH.reff

        # ACCRETION
        rho_rad = self.engine.rho_radiation(T_rad)
        if self.use_effective_radius:
            accretion_rate = self.accretion_efficiency_F * np.pi * reff*reff * rho_rad
        else:
            accretion_rate = self.accretion_efficiency_F * np.pi * R_H*R_H * rho_rad

        # EVAPORATION

        # compute black hole Hawking temperature
        T_H = M_PBH.T_Hawking

        # evaporation into on-brane degrees of freedom
        Area4D = Const_4Pi * reff*reff
        evaporation_rate4d = self.engine.params.gstar * self.engine.params.StefanBoltzmannConstant4D * Area4D * T_H*T_H*T_H*T_H

        reff5d = M_PBH.reff_5D
        Area5D = Const_2PiSquared * reff5d*reff5d*reff5d
        evaporation_rate5d = 5.0 * self.engine.params.StefanBoltzmannConstant5D * Area5D * T_H*T_H*T_H*T_H*T_H

        evaporation_rate = evaporation_rate4d + evaporation_rate5d

        dMdt = accretion_rate - evaporation_rate

        # convert dMdt into dMdT
        with np.errstate(all='raise'):
            try:
                dMdT = -dMdt / (Hubble*T_rad)
            except FloatingPointError as w:
                print('!! WARNING: {w}, dMdt = {dMdt:.5g}, T_rad = {Trad:.5g}, '
                      'H = {H:.5g}'.format(w = str(w), dMdt=dMdt, Trad=T_rad, H=Hubble))

        print('-- integrator returning dMdT = {dMdT:.5g}, dMdt = {dMdt:.5g}, accretion rate = {acc:.5g}, '
              'evaporation rate = {evap:.5g}, 4d evaporation = {evap4:.5g}, 5d evaporation = {evap5:.5g}' \
                .format(dMdT=dMdT, dMdt=dMdt, acc=accretion_rate, evap=evaporation_rate,
                        evap4=evaporation_rate4d, evap5=evaporation_rate5d))

        return dMdT


class PBH_lifetime_observer:
    # constructor captures CosmologyEngine instance. T_grid should be a numpy 1d array representing points where
    # we want to sample the solution M(T), and M_grid is an (empty) numpy 1d array of the same shape
    # into which the answer will be written
    def __init__(self, engine: CosmologyEngine, T_grid, M_grid):
        if engine is None or not isinstance(engine, CosmologyEngine):
            raise RuntimeError('PBH_lifetime_observer: supplied CosmologyEngine instance is not usable')

        if T_grid.shape != M_grid.shape:
            raise RuntimeError('PBH_lifetime_observer: T_grid and M_grid shapes do not match')

        self.engine = engine
        self.terminate_mass = engine.params.M4

        self.terminated = False

        self.T_grid = T_grid
        self.M_grid = M_grid

        self.T_grid_length = T_grid.size
        self.T_grid_current_index = 0
        if self.T_grid_length > 0:
            self.T_grid_next = self.T_grid[self.T_grid_current_index]
        else:
            self.T_grid_next = None

    # observation step should sample the solution if needed, and check whether the integration should end
    def obs(self, T_rad, M_PBH_asarray):
        M_PBH = M_PBH_asarray.item()
        print('-- observer being called at T_rad = {TradGeV:.5g} GeV = {TradKelvin:.5g} Kelvin, '
              'mass = {MassGeV:.5g} = {MassGrams:.5g} grams'.format(TradGeV=T_rad, TradKelvin=T_rad/Kelvin,
                                                                    MassGeV=M_PBH, MassGrams=M_PBH/Gram))

        # write solution into M-grid if we have passed an observation point
        if self.T_grid_next is not None and T_rad < self.T_grid_next:
            print('-- observer storing solution at index position {n}, T_rad = {TradGeV:.5g} GeV, '
                  'mass = {MassGeV:.5g}'.format(n=self.T_grid_current_index, TradGeV=T_rad, MassGeV=M_PBH))
            self.M_grid[self.T_grid_current_index] = M_PBH

            self.T_grid_current_index += 1
            if self.T_grid_current_index < self.T_grid_length:
                self.T_grid_next = self.T_grid[self.T_grid_current_index]
            else:
                self.T_grid_next = None

        # check whether integration should halt because we have decreased the PBH mass below the 4D Planck scale M4.
        # If this happens, we either get a relic, or at least the standard calculation of Hawking radiation is
        # invalidated, so either way we should stop
        if M_PBH < self.terminate_mass:
            self.terminated = True
            return -1

        return 0


# class PBHInstance captures details of a PBH that forms at a specified initial temperature
# (which we can map to an initial mass and a lengthscale)
class PBHInstance:
    # capture cosmology engine instance and formation temperature of the PBH, measured in GeV
    def __init__(self, engine: CosmologyEngine, T_init: float, horizon_mass_fraction_f=1.0,
                 num_sample_points=NumTSamplePoints):
        self.engine = engine
        self.T_rad_init = T_init

        # fraction of horizon mass that collapses to form the black hole
        self.horizon_mass_fraction_f = 1.0

        # map initial temperature to initial mass of the PBH, in this case for Randall-Sundrum
        self.M_init = horizon_mass_fraction_f * engine.M_Hubble(T_init)

        # also compute initial temperature for the standard 4D scenario
        self.M_init4 = horizon_mass_fraction_f * engine.M_Hubble4(T_init)

        # map iniial temperature to a length scale corresponding to the perturbation enclosing the initial mass,
        # here for Randall-Sundrum
        self.R_init = engine.R_Hubble(T_init)
        self.k_init = Const_2Pi / self.R_init

        # and also for the standard 4D scenario
        self.R_init4 = engine.R_Hubble4(T_init)
        self.k_init4 = Const_2Pi / self.R_init4

        # set up grid of T and M sample points to
        # record this mass/temperature relation and the corresponding lifetime;
        self.T_sample_points = np.geomspace(T_init, T_CMB * Kelvin, num_sample_points)

        # compute baseline lifetime for this PBH, that is: in Randall-Sundrum scenario, using
        # effective radius and greybody factors
        self.lifetime = self.compute_PBH_lifetime(use_effective_radius=True,
                                                  use_greybody_factors=True)

    # compute the PBH lifetime (expressed as a final temperature in GeV),
    # given an initial PBH mass (expressed in GeV) and an initial temperature for the
    # radiation bath (expressed in GeV)
    def compute_PBH_lifetime(self, use_effective_radius=True, use_greybody_factors=True):
        # set up ODE system to integrate, select an integrator, and set initial conditions
        rhs = PBH_lifetime_RHS(self.engine,
                               accretion_efficiency_F=1.0,
                               use_effective_radius=use_effective_radius,
                               use_greybody_factors=use_greybody_factors)

        # set up stepper; need to use on that supports solout, which the SUNDIALS ones don't seem to do
        stepper = ode(rhs.eval).set_integrator('dop853', rtol=1E-6, nsteps=5000)

        # set up an observer to detect when the integration should terminate
        M_sample_points = np.zeros_like(self.T_sample_points)
        observer = PBH_lifetime_observer(self.engine, self.T_sample_points, M_sample_points)
        stepper.set_solout(observer.obs)

        # set up initial conditions for the PBH and the radiation bath
        # to keep the numerics sensible, we can't run the integration directly in grams; the numbers get too large,
        # making the integrator need a very small stepsize to keep up
        stepper.set_initial_value(self.M_init, self.T_rad_init)

        # integrate down to the present CMB temperature, or when the observer notices that the PBH
        # mass has decreased below M4
        T_min = T_CMB * Kelvin

        while stepper.successful() and observer.T_grid_next is not None and stepper.t > T_min \
                and not observer.terminated:
            # work out the current target time
            # this is taken to be the current time minus 1/10 of the distance of the next sample point,
            # or the current time minus 5% of the next sample point, whichever is larger.
            # The idea is that we don't run for too long, so we don't exceed the integrator's internal
            # step limit nsteps
            Tdiff = stepper.t - observer.T_grid_next
            Tdelta = Tdiff / 10.0
            if Tdelta < 0.05 * observer.T_grid_next:
                Tdelta = 0.05 * observer.T_grid_next
            Ttarget = stepper.t - Tdelta
            print('%% trying to integrate to T = {Tnext}'.format(Tnext=Ttarget))
            stepper.integrate(Ttarget)

        # if there was an integration failure, raise an exception
        if not stepper.successful():
            raise RuntimeError('PBH lifetime calculation failed due to an integration error, '
                               'code = {code}'.format(code=stepper.get_return_code()))

        # truncate unused sample points at end of M_sample_points
        M_sample_points.resize(observer.T_grid_current_index)

        # if the PBH mass has decreased to the point where we get a relic, report the lifetime that was found
        if observer.terminated or stepper.y <= self.engine.params.M4:
            return stepper.T, M_sample_points

        # A lifetime of None indicates that this PBH has not evaporated by the present day
        return None, M_sample_points

    def lifetime_plot(self, filename, mass_units='gram', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in ['gram', 'kilogram', 'GeV']:
            raise RuntimeError('lifetime_plot(): unit "{unit}" not understood in constructor'.format(unit=mass_units))

        if temperature_units not in ['Kelvin', 'GeV']:
            raise RuntimeError(
                'lifetime_plot: unit "{unit}" not understood in constructor'.format(unit=temperature_units))

        mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}
        temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}

        mass_unit = mass_conversions[mass_units]
        temperature_unit = temperature_conversions[temperature_units]

        T_values = self.T_sample_points[:self.lifetime[1].size] / temperature_unit
        M_values = self.lifetime[1] / mass_unit

        plt.figure()
        plt.loglog(T_values, M_values, label='initial mass = {i:.3g} {unit}'.format(i=self.M_init / mass_unit,
                                                                                    unit=mass_units))
        plt.xlabel('Temperature T / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH mass / {unit}'.format(unit=mass_units))
        plt.legend()
        plt.savefig(filename)


# generate a plot of PBH formation mass vs. formation temperature
def PBHMassPlot(M5, Tlo=1E3, Thi=None, units='gram'):
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

    M_values = [engine.M_Hubble(T) / unit for T in T_range]
    M4_values = [engine.M_Hubble4(T) / unit for T in T_range]

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

    R_values = [engine.R_Hubble(T) / unit for T in T_range]
    R4_values = [engine.R_Hubble4(T) / unit for T in T_range]

    plt.figure()
    plt.loglog(T_range, R_values, label='Randall-Sundrum')
    plt.loglog(T_range, R4_values, label='Standard 4D')
    plt.xlabel('Temperature T / GeV')
    plt.ylabel('Collapse lengthscale / {units}'.format(units=units))
    plt.legend()
    plt.savefig('formation-lengthscale.pdf')


# generate a plot of PBH formation mass vs. formation lengthscale
def PBHMassScaleRelation(M5, Tlo=4E3, Thi=None, length_units='kilometre', mass_units='gram'):
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

    R_values = [engine.R_Hubble(T) / length_unit for T in reversed(T_range)]
    R4_values = [engine.R_Hubble4(T) / length_unit for T in reversed(T_range)]

    M_values = [engine.M_Hubble(T) / mass_unit for T in reversed(T_range)]
    M4_values = [engine.M_Hubble4(T) / mass_unit for T in reversed(T_range)]

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
engine = CosmologyEngine(params)
sample = PBHInstance(engine, 1E10)
sample.lifetime_plot('PBH_T1E10_lifetime.pdf')
