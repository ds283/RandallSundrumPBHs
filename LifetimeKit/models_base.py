import math
from operator import itemgetter
from abc import ABC, abstractmethod

import numpy as np

from .constants import Page_suppression_factor
from .natural_units import Kelvin
from .particle_data import SM_particle_table

# tolerance for binning particle masses (expressed in GeV) into a single threshold
T_threshold_tolerance = 1E-8

Const_2Pi = 2.0 * math.pi
Const_4Pi = 4.0 * math.pi
Const_PiOver2 = math.pi / 2.0

def build_cumulative_g_table(particle_table, weight=None):
    # build a list of particle records ordered by their mass
    particles = list(particle_table.values())
    particles.sort(key=itemgetter('mass'))

    next_threshold = 0.0
    cumulative_g = 0.0
    current_threshold_index = 0
    current_particle_index = 0

    num_particles = len(particles)
    T_thresholds = np.zeros(num_particles)
    g_values = np.zeros(num_particles)

    next_record = particles[current_particle_index]
    while current_particle_index < num_particles:
        # reset current_threshold to value of next mass threshold
        next_threshold = next_record['mass']

        while current_particle_index < num_particles:
            next_record = particles[current_particle_index]
            if next_record['mass'] - next_threshold > T_threshold_tolerance:
                break

            cumulative_g += next_record['dof'] * (next_record[weight] if weight is not None else 1.0)

            current_particle_index += 1

        T_thresholds[current_threshold_index] = next_threshold
        g_values[current_threshold_index] = cumulative_g
        current_threshold_index += 1

    T_thresholds = np.resize(T_thresholds, current_threshold_index)
    g_values = np.resize(g_values, current_threshold_index)

    return T_thresholds, g_values


class BaseCosmology:

    def __init__(self, params, fixed_g=None):
        self.params = params

        self.SM_thresholds, self.SM_g_values = build_cumulative_g_table(SM_particle_table, weight='spin-weight')
        self.SM_num_thresholds = len(self.SM_thresholds)

        self._fixed_g = fixed_g

    # compute the radiation energy density in GeV^4 from a temperature supplied in GeV
    # currently, we assume there are a fixed number of relativistic species
    def rho_radiation(self, T=None, log_T=None):
        # if T is not supplied, try to use log_T
        if T is not None:
            _T = T
        elif log_T is not None:
            _T = math.exp(log_T)
        else:
            raise RuntimeError('No temperature value supplied to BaseCosmology.rho_radiation()')

        # check that supplied temperature is lower than the intended maximum temperature
        # (usually the 4D or 5D Planck mass, but model-dependent)
        if T > self.params.Tmax:
            raise RuntimeError('Temperature T = {TGeV:.3g} GeV = {TKelvin:.3g} K is higher than the specified '
                               'maximum temperature Tmax = {TmaxGeV:.3g} GeV = '
                               '{TmaxKelvin:.3g} K'.format(TGeV=T, TKelvin=T/Kelvin,
                                                           TmaxGeV=self.params.Tmax, TmaxKelvin=self.params.Tmax/Kelvin))

        Tsq = T*T
        T4 = Tsq*Tsq

        if self._fixed_g:
            g = self._fixed_g
        else:
            g = self.g(T)

        return self.params.RadiationConstant * g * T4


    def g(self, T):
        """
        Compute number of relativistic degrees of freedom in the radiation bath at temperature T
        """

        # find where radiation temperature lies within out threshold list
        index = np.searchsorted(self.SM_thresholds, T, side='left')
        if index >= self.SM_num_thresholds:
            index = self.SM_num_thresholds - 1

        return self.SM_g_values[index]


class BondiHoyleLyttletonAccretionModel:
    """
    Shared implementation used by all instances of Bondi-Hoyle-Lyttleton-type
    accretion models
    """
    def __init__(self, engine, accretion_efficiency_F, use_effective_radius, use_Page_suppression):
        self.engine = engine

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

    def rate(self, T_rad, PBH):
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        dM_dt = math.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho

        # if using Page suppression, divide by Page suppression factor since presumably
        # this affects accretion equally to radiation by a detailed balance argument
        # in equilibrium
        return dM_dt / (Page_suppression_factor if self._use_Page_suppression else 1.0)


class StefanBoltzmann4D:
    """
    Shared implementation of Stefan-Boltzmann law in 4D
    """

    def __init__(self, SB_4D: float, use_effective_radius=True, use_Page_suppression=True):
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        self._SB_4D = SB_4D

    def rate(self, PBH, g4=1.0):
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        t = PBH.t
        t4 = t*t*t*t

        evap_prefactor = Const_4Pi * alpha_sq / (t4 * rh_sq)
        evap_dof = g4 * self._SB_4D

        dM_dt = -evap_prefactor * evap_dof / (Page_suppression_factor if self._use_Page_suppression else 1.0)

        return dM_dt


class StefanBoltzmann5D:
    """
    Shared implementation of Stefan-Boltzmannn law in 5D
    """

    def __init__(self, SB_4D: float, SB_5D: float, use_effective_radius=True, use_Page_suppression=True):
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        self._SB_4D = SB_4D
        self._SB_5D = SB_5D

    def dMdt(self, PBH, g4=0.0, g5=1.0):
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        t = PBH.t
        t4 = t*t*t*t

        try:
            evap_prefactor = Const_4Pi * alpha_sq / (t4 * rh_sq)
            evap_dof = (g4 * self._SB_4D + Const_PiOver2 * alpha * g5 * self._SB_5D / t)

            dM_dt = -evap_prefactor * evap_dof / (Page_suppression_factor if self._use_Page_suppression else 1.0)
        except ZeroDivisionError:
            dM_dt = float("nan")

        return dM_dt

class BaseStefanBoltzmannLifetimeModel:
    """
    Shared infrastructure used by all Stefan-Boltzmann-type lifetime models
    """

    def __init__(self, engine, Model, BlackHole, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 extra_4D_state_table=None):
        """
        To speed up computations, we want to cache the number of relativistic degrees of freedom
        available at any given temperature.
        To do that we need to build a list of mass thresholds
        :param engine: a model engine instance to use for computations
        :param Model: expected type of engine
        :param BlackHole: type of BlackHole instance object
        :param accretion_efficiency_F:
        :param use_effective_radius:
        :param use_Page_suppression:
        """
        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('BaseStefanBoltzmannLifetimeModel: supplied engine instance is not of expected type')

        self.engine = engine
        self._params = engine.params

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        # create an accretion model instnce
        self._accretion_model = \
            BondiHoyleLyttletonAccretionModel(engine, accretion_efficiency_F, use_effective_radius,
                                              use_Page_suppression)

        # create a PBHModel instance; the value assigned to the mass doesn't matter
        self._PBH = BlackHole(self.engine.params, 1.0, units='gram')

        # build table of mass thresholds associated with Hawking quanta; this is used in the Stefan-Boltzmann
        # approximation
        particle_table = SM_particle_table

        # extra particle states allow us to add extra dofs to the Standard Model ones, if needed,
        # when calculating the number of degrees of freedom available for emission into Hawking quanta.
        # These are only used in the Stefan-Boltzmann approximation.
        #
        # In a Randall-Sundrum model we need to account separately for emission of gravitons into the bulk
        if extra_4D_state_table is not None:
            particle_table = particle_table | extra_4D_state_table

        self._thresholds, self._g_values = build_cumulative_g_table(particle_table)
        self._num_thresholds = len(self._thresholds)


    def g4(self, T_Hawking):
        """
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into,
        based on Standard Model particles
        """

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self._thresholds, T_Hawking, side='left')
        if index >= self._num_thresholds:
            index = self._num_thresholds - 1

        return self._g_values[index]

    def _dMdt_accretion(self, T_rad, PBH):
        return self._accretion_model.rate(T_rad, PBH)


class BaseGreybodyLifetimeModel(ABC):
    """
    Base infrastructure used by all greybody-factor type lifetime models
    """

    def __init__(self, engine, Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True):
        """
        :param engine: a model engine instance to use for computations
        :param Model: expected type of engine
        :param accretion_efficiency_F:
        :param use_effective_radius:
        :param use_Page_suppression:
        """
        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('BaseFriedlanderGreybodyLifetimeModel: supplied engine instance is not of expected type')

        self.engine = engine
        self._params = engine.params

        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        # create an accretion model instnce
        self._accretion_model = \
            BondiHoyleLyttletonAccretionModel(engine, accretion_efficiency_F, use_effective_radius,
                                              use_Page_suppression)

    @abstractmethod
    def xi_species_list(self, PBH):
        """
        Virtual method that should return a dictionary with keys corresponding to the different particle
        species labels
        :param PBH: needed to allow switching between different xi values depending on whether the PBH
        is in a 4D or 5D regime
        """
        pass

    def _sum_dMdt_species(self, PBH, species: str):
        """
        compute dM/dt for the species labels in 'species', for the black hole parameter configurations
        in 'PBH'
        :param PBH: current black hole object, used to abstract Hawking temperature calculation
        :param species: list of names of species to include
        :return: emission rate in Mass/Time = [Energy]^2
        """
        rh = PBH.radius
        rh_sq = rh*rh

        # compute Hawking temperature
        T_Hawking = PBH.T_Hawking

        # cache reference to xi-table dictionary
        xi_species_list = self.xi_species_list(PBH)

        dM_dt = 0.0
        for label in species:
            record = xi_species_list[label]

            # If record is a callable, then it is a partially evaluated greybody function that requires
            # to be evaluated on the current Hawking temperature. This happens when the species is massive,
            # because then its emission rate is temperature dependent.
            # On the other hand, if record is not callable, it is a pre-evaluated emission rate.
            # This happens when the species is massless, because then its emission rate is not temperature
            # dependent.
            if callable(record):
                dM_dt += record(T_Hawking)
            else:
                dM_dt += record

        return -dM_dt / (Const_2Pi * rh_sq)

    def _dMdt_accretion(self, T_rad, PBH):
        """
        Compute dM/dt rate from accretion
        :param T_rad: radiation temperature in GeV
        :PBH: object representing PBH
        """
        return self._accretion_model.rate(T_rad, PBH)

    def _dMdt_quarks(self, T_rad, PBH):
        """
        Compute dM/dt rate from emission into quarks
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        quarks = ['up quark', 'down quark', 'strange quark', 'charm quark', 'bottom quark', 'top quark']
        return self._sum_dMdt_species(PBH, quarks)

    def _dMdt_leptons(self, T_rad, PBH):
        """
        Compute dM/dt rate from emission into leptons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        leptons = ['electron', 'muon', 'tau', 'neutrino']
        return self._sum_dMdt_species(PBH, leptons)

    def _dMdt_photons(self, T_rad, PBH):
        """
        Compute dM/dt rate from emission into photons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dMdt_species(PBH, ['photon'])

    def _dMdt_gluons(self, T_rad, PBH):
        """
        Compute dM/dt rate from emission into gluons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dMdt_species(PBH, ['gluon'])

    def _dMdt_EW_bosons(self, T_rad, PBH):
        """
        Compute dM/dt rate from emission into electroweak bosons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        EW_bosons = ['Higgs', 'Z boson', 'W boson']
        return self._sum_dMdt_species(PBH, EW_bosons)


class BaseFriedlanderGreybodyLifetimeModel(BaseGreybodyLifetimeModel):
    """
    Base infrastructure used by all greybody models using Friedlander et al. fitting functions
    """

    def __init__(self, engine, Model, BlackHole, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True):
        """
        :param engine: a model engine instance to use for computations
        :param Model: expected type of engine
        :param BlackHole: type of BlackHole instance object
        :param accretion_efficiency_F:
        :param use_effective_radius:
        :param use_Page_suppression:
        """
        super().__init__(engine, Model, accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius, use_Page_suppression=use_Page_suppression)

        # create a PBHModel instance; for a Friedlander et al. model, this just has mass but no angular
        # moementum the value assigned to the mass doesn't matter
        self._PBH = BlackHole(self.engine.params, 1.0, units='gram')

    @abstractmethod
    def massless_xi(self, PBH):
        """
        All greybody models have a self.xi_species_list() method that returns a dictionary with keys
        corresponding to the particle species labels, and elements corresponding to xi values (either constants
        if there is no temperature dependence, or callables if the temperature dependence is needed).

        However, for Friedlander et al. models we use an extra layer of caching, with all temperature-dependent
        xi values consolidated into self.massive_xi() (which is a list of callables) and all temperature-independent
        xi values consolidated into a single float self.massless_xi(). This means we don't have to iterate
        through a dictionary and is supposed to speed up evaluation.

        As for self.xi_species_list(), there is a PBH argument because implementations may wish to vary
        their behaviour depending whether the black hole is in a 4D or 5D regime (or perhaps based on other
        parameters also)
        :param PBH: PBH object that can be interrogated to determine parameters
        """
        pass

    @abstractmethod
    def massive_xi(self, PBH):
        """
        All greybody models have a self.xi_species_list() method that returns a dictionary with keys
        corresponding to the particle species labels, and elements corresponding to xi values (either constants
        if there is no temperature dependence, or callables if the temperature dependence is needed).

        However, for Friedlander et al. models we use an extra layer of caching, with all temperature-dependent
        xi values consolidated into self.massive_xi() (which is a list of callables) and all temperature-independent
        xi values consolidated into a single float self.massless_xi(). This means we don't have to iterate
        through a dictionary and is supposed to speed up evaluation.

        As for self.xi_species_list(), there is a PBH argument because implementations may wish to vary
        their behaviour depending whether the black hole is in a 4D or 5D regime (or perhaps based on other
        parameters also)
        :param PBH: PBH object that can be interrogated to determine parameters
        """
        pass
