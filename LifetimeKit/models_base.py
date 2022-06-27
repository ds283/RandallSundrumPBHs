import math
from abc import ABC, abstractmethod
from operator import itemgetter

import numpy as np
from scipy.optimize import brentq

from .constants import Page_suppression_factor
from .natural_units import Kelvin, Kilogram, Gram, SolarMass
from .particle_data import SM_particle_table

# tolerance for binning particle masses (expressed in GeV) into a single threshold
_T_threshold_tolerance = 1E-8

_Const_2Pi = 2.0 * math.pi
_Const_4Pi = 4.0 * math.pi
_Const_PiOver2 = math.pi / 2.0

# minimum temperature for which we search when trying to compute a T_init given
# an M_init. For now, we take this to be TeV = 1E3 GeV
# The answer is expressed in GeV
_MINIMUM_SEARCH_TEMPERATURE = 1E3

_TEMPERATURE_COMPARE_TOLERANCE = 1E-6

_ROOT_SEARCH_TOLERANCE = 1E-3

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
            if next_record['mass'] - next_threshold > _T_threshold_tolerance:
                break

            cumulative_g += next_record['dof'] * (next_record[weight] if weight is not None else 1.0)

            current_particle_index += 1

        T_thresholds[current_threshold_index] = next_threshold
        g_values[current_threshold_index] = cumulative_g
        current_threshold_index += 1

    T_thresholds = np.resize(T_thresholds, current_threshold_index)
    g_values = np.resize(g_values, current_threshold_index)

    return T_thresholds, g_values


class BaseCosmology(ABC):
    """
    Defines interface and common infrastructure for a cosmology engine
    """
    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'SolarMass': SolarMass, 'GeV': 1.0}

    def __init__(self, params, fixed_g=None) -> None:
        super().__init__()
        self._params = params

        self.SM_thresholds, self.SM_g_values = build_cumulative_g_table(SM_particle_table, weight='spin-weight')
        self.SM_num_thresholds = len(self.SM_thresholds)

        self._fixed_g = fixed_g

    def rho_radiation(self, T: float=None, log_T: float=None) -> float:
        """
        Compute the radiation energy density in GeV^4 from a temperature supplied in GeV.
        Currently, we assume there are a fixed number of relativistic species
        :param T:
        :param log_T:
        :return:
        """
        # if T is not supplied, try to use log_T
        if T is not None:
            _T = T
        elif log_T is not None:
            _T = math.exp(log_T)
        else:
            raise RuntimeError('No temperature value supplied to BaseCosmology.rho_radiation()')

        # check that supplied temperature is lower than the intended maximum temperature
        # (usually the 4D or 5D Planck mass, but model-dependent)
        if _T/self._params.Tmax > 1.0 + _TEMPERATURE_COMPARE_TOLERANCE:
            raise RuntimeError('Temperature T = {TGeV:.3g} GeV = {TKelvin:.3g} K is higher than the specified '
                               'maximum temperature Tmax = {TmaxGeV:.3g} GeV = '
                               '{TmaxKelvin:.3g} K'.format(TGeV=_T, TKelvin=_T/Kelvin,
                                                           TmaxGeV=self._params.Tmax, TmaxKelvin=self._params.Tmax / Kelvin))

        Tsq = _T*_T
        T4 = Tsq*Tsq

        if self._fixed_g:
            g = self._fixed_g
        else:
            g = self.g(_T)

        return self._params.RadiationConstant * g * T4


    def g(self, T: float) -> float:
        """
        Compute number of relativistic degrees of freedom in the radiation bath at temperature T
        """

        # find where radiation temperature lies within out threshold list
        index = np.searchsorted(self.SM_thresholds, T, side='left')
        if index >= self.SM_num_thresholds:
            index = self.SM_num_thresholds - 1

        return self.SM_g_values[index]

    @abstractmethod
    def Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
        :param T:
        :param log_T:
        :return:
        """
        pass

    @abstractmethod
    def R_Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
        the formula here is R_H = 1/H
        :param T:
        :param log_T:
        :return:
        """
        pass

    @abstractmethod
    def M_Hubble(self, T: float=None, log_T: float=None) -> float:
        """
        compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
        the formula here is M_H = (4/3) pi rho R_H^3, but we compute it directly to avoid multiple evaluations of rho
        :param T:
        :param log_T:
        :return:
        """
        pass

    def find_Tinit_from_Minit(self, M: float, units: str='GeV') -> float:
        """
        solve to find the initial radiation temperature that corresponds to a given
        initial PBH mass M_init
        :param M:
        :param units:
        :return:
        """
        if units not in self._mass_conversions:
            raise RuntimeError('BaseCosmology.find_Tinit_from_Minit: unit "{unit}" not understood in constructor'.format(unit=units))

        units_to_GeV = self._mass_conversions[units]
        M_target = M * units_to_GeV
        log_M_target = math.log(M_target)

        # try to bracket the root within a sensible range of temperature
        log_T_max = math.log(self._params.Tmax)
        log_T_min = math.log(_MINIMUM_SEARCH_TEMPERATURE)

        def log_M_Hubble_root(log_T: float) -> float:
            return math.log(self.M_Hubble(log_T=log_T)) - log_M_target

        log_T_sample = np.linspace(start=log_T_min, stop=log_T_max, num=50)
        # print('log_T_min = {min}, log_T_max = {max}, T_max = {Tmax}'.format(min=log_T_min, max=log_T_max, Tmax=self._params.Tmax))

        root_sample = np.array(list(map(log_M_Hubble_root, log_T_sample)), dtype=np.float64)

        # print('M_target = {Mtarget} gram'.format(Mtarget=M_target/Gram))
        # print('log_T_sample = {Tsamp}'.format(Tsamp=log_T_sample))
        # print('log_M_sample = {logMsamp}'.format(logMsamp=log_M_sample))
        # print('root_sample = {root}'.format(root=root_sample))

        # find where elements of M_sample change sign
        sign_changes = np.where(np.diff(np.sign(root_sample)) != 0)[0] + 1
        # print('sign_changes = {changes}'.format(changes=sign_changes))

        if sign_changes.size == 0:
            raise RuntimeError('BaseCosmology.find_Tinit_from_Minit: found no roots')
        if sign_changes.size > 1:
            raise RuntimeError('BaseCosmology.find_Tinit_from_Minit: found multiple roots')

        sign_change_point = sign_changes[0]
        a = log_T_sample[sign_change_point-1]
        b = log_T_sample[sign_change_point]
        if log_M_Hubble_root(a)*log_M_Hubble_root(b) > 0.0:
            raise RuntimeError('BaseCosmology.find_Tinit_from_Minit: extrema did not bracket a root')

        found_logT = brentq(log_M_Hubble_root, a=a, b=b)
        found_T = math.exp(found_logT)
        found_M = self.M_Hubble(log_T=found_logT)

        found_vs_sought = found_M/M_target
        if math.fabs(found_vs_sought - 1.0) > _ROOT_SEARCH_TOLERANCE:
            raise RuntimeError('BaseCosmology.find_Tinit_from_Minit: target M = {Mtarget} gram, '
                               'found M = {Mfound} gram, found T = {Tfound} K'.format(Mtarget=M_target / Gram,
                                                                                      Mfound=found_M / Gram,
                                                                                      Tfound=found_T / Kelvin))

        return found_T


class BondiHoyleLyttletonAccretionModel:
    """
    Shared implementation used by all instances of Bondi-Hoyle-Lyttleton-type
    accretion models
    """
    def __init__(self, engine, accretion_efficiency_F, use_effective_radius, use_Page_suppression) -> None:
        self.engine = engine

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

    def rate(self, T_rad: float, PBH) -> float:
        """
        Compute accretion rate (in GeV^2) for a given PBH instance at a given radiation temperature (in GeV)
        :param T_rad:
        :param PBH:
        :return:
        """
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

    def __init__(self, SB_4D: float, use_effective_radius=True, use_Page_suppression=True) -> None:
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        self._SB_4D = SB_4D

    def rate(self, PBH, g4: float=1.0) -> float:
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        t = PBH.t
        t4 = t*t*t*t

        evap_prefactor = _Const_4Pi * alpha_sq / (t4 * rh_sq)
        evap_dof = g4 * self._SB_4D

        dM_dt = -evap_prefactor * evap_dof / (Page_suppression_factor if self._use_Page_suppression else 1.0)

        return dM_dt


class StefanBoltzmann5D:
    """
    Shared implementation of Stefan-Boltzmannn law in 5D
    """

    def __init__(self, SB_4D: float, SB_5D: float, use_effective_radius=True, use_Page_suppression=True) -> None:
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        self._SB_4D = SB_4D
        self._SB_5D = SB_5D

    def dMdt(self, PBH, g4: float=0.0, g5: float=1.0) -> float:
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        t = PBH.t
        t4 = t*t*t*t

        try:
            evap_prefactor = _Const_4Pi * alpha_sq / (t4 * rh_sq)
            evap_dof = (g4 * self._SB_4D + _Const_PiOver2 * alpha * g5 * self._SB_5D / t)

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
                 extra_4D_state_table=None) -> None:
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


    def g4(self, T_Hawking: float) -> float:
        """
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into,
        based on Standard Model particles
        """

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self._thresholds, T_Hawking, side='left')
        if index >= self._num_thresholds:
            index = self._num_thresholds - 1

        return self._g_values[index]

    def _dMdt_accretion(self, T_rad: float, PBH) -> float:
        return self._accretion_model.rate(T_rad, PBH)


class BaseGreybodyLifetimeModel(ABC):
    """
    Base infrastructure used by all greybody-factor type lifetime models
    """

    def __init__(self, engine, Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True) -> None:
        """
        :param engine: a model engine instance to use for computations
        :param Model: expected type of engine
        :param accretion_efficiency_F:
        :param use_effective_radius:
        :param use_Page_suppression:
        """
        super().__init__()
        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('BaseFriedlanderGreybodyLifetimeModel: supplied engine instance is not of expected type')

        self.engine = engine
        self._params = engine._params

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

    @abstractmethod
    def _sum_dMdt_species(self, PBH, species: str) -> float:
        """
        compute dM/dt for the species labels in 'species', for the black hole parameter configurations
        in 'PBH'
        :param PBH: current black hole object, used to abstract Hawking temperature calculation
        :param species: list of names of species to include
        :return: emission rate in Mass/Time = [Energy]^2
        """
        pass

    def _dMdt_accretion(self, T_rad: float, PBH) -> float:
        """
        Compute dM/dt rate from accretion
        :param T_rad: radiation temperature in GeV
        :PBH: object representing PBH
        """
        return self._accretion_model.rate(T_rad, PBH)

    def _dMdt_quarks(self, T_rad: float, PBH) -> float:
        """
        Compute dM/dt rate from emission into quarks
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        quarks = ['up quark', 'down quark', 'strange quark', 'charm quark', 'bottom quark', 'top quark']
        return self._sum_dMdt_species(PBH, quarks)

    def _dMdt_leptons(self, T_rad: float, PBH) -> float:
        """
        Compute dM/dt rate from emission into leptons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        leptons = ['electron', 'muon', 'tau', 'neutrino']
        return self._sum_dMdt_species(PBH, leptons)

    def _dMdt_photons(self, T_rad: float, PBH) -> float:
        """
        Compute dM/dt rate from emission into photons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dMdt_species(PBH, ['photon'])

    def _dMdt_gluons(self, T_rad: float, PBH) -> float:
        """
        Compute dM/dt rate from emission into gluons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dMdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dMdt_species(PBH, ['gluon'])

    def _dMdt_EW_bosons(self, T_rad: float, PBH) -> float:
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
                 use_effective_radius=True, use_Page_suppression=True) -> None:
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

        # create a PBH model instance; for a Friedlander et al. model, this just has mass but no angular
        # moementum the value assigned to the mass doesn't matter
        self._PBH = BlackHole(self.engine._params, 1.0, units='gram')

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

    def _sum_dMdt_species(self, PBH, species: str) -> float:
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

        return -dM_dt / (_Const_2Pi * rh_sq)

    def _dMdt_evaporation(self, T_rad: float, PBH) -> float:
        """
        Compute evaporation rate for a specified black hole configuration.
        Note that the radiation temperature is supplied as T_rad, although it is not used; this is
        because all _dMdt_* methods need to have the same signature, and the _dMdt_accretion() method
        *does* need to know the radiation temperature.
        :param T_rad:
        :param PBH:
        :return:
        """
        # compute horizon radius in 1/GeV
        rh = PBH.radius
        rh_sq = rh*rh

        # compute Hawking temperature
        T_Hawking = PBH.T_Hawking

        # cache tables of massless and massive xi values
        massless_xi = self.massless_xi(PBH)
        massive_xi = self.massive_xi(PBH)

        # sum over xi factors to get evaporation rate
        try:
            dM_dt = -(massless_xi + sum([xi(T_Hawking) for xi in massive_xi])) / (_Const_2Pi * rh_sq)
        except ZeroDivisionError:
            dM_dt = float("nan")

        return dM_dt


class BaseSpinningGreybodyLifetimeModel(BaseGreybodyLifetimeModel):
    """
    Base infrastructure used by all "spinning" black hole lifetime models that use the massless
    BlackMax/Kerr greybody xi values
    """

    def __init__(self, engine, Model, BlackHole, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True) -> None:
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

        # create a PBH model instance; for this type of black hole, we expect it to have both
        # mass and angular momentum
        self._PBH = BlackHole(self.engine.params, 1.0, J=0.0, units='gram')

    def _sum_dMdt_species(self, PBH, species: str) -> float:
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

        # for the 5D Randall-Sundrum black hole, this will switch between the Kerr a* = J/Jmax and the Myers-Perry
        # a* = a/Rh as needed
        xi_astar_arg = PBH.xi_astar_argument

        # cache reference to xi-table dictionary
        xi_species_list = self.xi_species_list(PBH)

        dM_dt = 0.0
        try:
            for label in species:
                data = xi_species_list[label]

                # if the particle mass is larger than the current temperature, we switch off the
                # Hawking flux into this particle species
                mass = data['mass']
                if mass > T_Hawking:
                    continue

                g = data['dof'] if data['xi-per-dof'] else 1.0
                xi_M = data['xi_M']
                dM_dt += -g * xi_M(xi_astar_arg)
        except ZeroDivisionError:
            return float("nan")

        return dM_dt / (_Const_2Pi * rh_sq)

    def _sum_dJdt_species(self, PBH, species: str) -> float:
        """
        compute dJ/dt for the species labels in 'species', for the black hole parameter configurations
        in 'PBH'
        :param PBH: current black hole object, used to abstract Hawking temperature calculation
        :param species: list of names of species to include
        :return: emission rate in Mass/Time = [Energy]^2
        """
        rh = PBH.radius

        # compute Hawking temperature
        T_Hawking = PBH.T_Hawking

        # for the 5D Randall-Sundrum black hole, this will switch between the Kerr a* = J/Jmax and the Myers-Perry
        # a* = a/Rh as needed
        xi_astar_arg = PBH.xi_astar_argument

        # cache reference to xi-table dictionary
        # in the 5D Randall-Sundrum spinning black hole, this will switch between the 4D and 5D tables
        # depending on the current 4D vs 5D state
        xi_species_list = self.xi_species_list(PBH)

        dJ_dt = 0.0
        try:
            for label in species:
                data = xi_species_list[label]

                # if the particle mass is larger than the current temperature, we switch off the
                # Hawking flux into this particle species
                mass = data['mass']
                if mass > T_Hawking:
                    continue

                g = data['dof'] if data['xi-per-dof'] else 1.0
                xi_J = data['xi_J']
                dJ_dt += -g * xi_J(xi_astar_arg)
        except ZeroDivisionError:
            return float("nan")

        return dJ_dt / (_Const_2Pi * rh)

    def _dMdt_evaporation(self, T_rad: float, PBH) -> float:
        """
        Compute evaporation rate for a specified black hole configuration.
        Note that the radiation temperature is supplied as T_rad, although it is not used; this is
        because all _dMdt_* methods need to have the same signature, and the _dMdt_accretion() method
        *does* need to know the radiation temperature.
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dMdt_species(PBH, self.xi_species_list(PBH).keys())

    def _dJdt_evaporation(self, T_rad: float, PBH) -> float:
        """
        Compute evaporation rate for a specified black hole configuration.
        Note that the radiation temperature is supplied as T_rad, although it is not used; this is
        because all _dMdt_* methods need to have the same signature, and the _dMdt_accretion() method
        *does* need to know the radiation temperature.
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dJdt_species(PBH, self.xi_species_list(PBH).keys())

    def _dJdt_quarks(self, T_rad: float, PBH) -> float:
        """
        Compute dJ/dt rate from emission into quarks
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dJdt_
        functions)
        :PBH: object representing PBH
        """
        quarks = ['up quark', 'down quark', 'strange quark', 'charm quark', 'bottom quark', 'top quark']
        return self._sum_dJdt_species(PBH, quarks)

    def _dJdt_leptons(self, T_rad: float, PBH) -> float:
        """
        Compute dJ/dt rate from emission into leptons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dJdt_
        functions)
        :PBH: object representing PBH
        """
        leptons = ['electron', 'muon', 'tau', 'neutrino']
        return self._sum_dJdt_species(PBH, leptons)

    def _dJdt_photons(self, T_rad: float, PBH) -> float:
        """
        Compute dJ/dt rate from emission into photons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dJdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dJdt_species(PBH, ['photon'])

    def _dJdt_gluons(self, T_rad: float, PBH) -> float:
        """
        Compute dJ/dt rate from emission into gluons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dJdt_
        functions)
        :PBH: object representing PBH
        """
        return self._sum_dJdt_species(PBH, ['gluon'])

    def _dJdt_EW_bosons(self, T_rad: float, PBH) -> float:
        """
        Compute dJ/dt rate from emission into electroweak bosons
        :param T_rad: radiation temperature in GeV (needed because there is a universal signature for all _dJdt_
        functions)
        :PBH: object representing PBH
        """
        EW_bosons = ['Higgs', 'Z boson', 'W boson']
        return self._sum_dJdt_species(PBH, EW_bosons)


class BaseBlackHole(ABC):
    """
    This is an abstract base class that specifies the minimal interface a black hole model should provide.
    """

    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'GeV': 1.0}

    def __init__(self, params, M: float, units='GeV', strict=True) -> None:
        """
        capture basic details about the PBH model, including a parameters object for the cosmology in which
        it is living, and the mass M.
        Other properties such as angular momentum that not all black holes share should be captured
        in derived classes.
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param units: units used to measure the black hole mass
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__()
        self.params = params

        # assign current mass value in GeV
        # define a 'None' value first, in order to define all instance attributes within __init__()
        self.M: float = None
        self.set_M(M, units)

        # check mass is positive
        if self.M < 0.0:
            raise RuntimeError('BaseBlackHole: Initial black hole mass should be positive (M={M} GeV)'.format(M=M))

        # check mass is larger than 4D Planck mass (has to be done *after* assignment so that any units
        # conversions have taken place)
        if strict and self.M <= params.M4:
            raise RuntimeError('BaseBlackHole: Initial black hole mass {mass} GeV should be larger than the '
                               '4D Planck mass {MP} GeV in order that the PBH does not begin life as a '
                               'relic'.format(mass=self.M, MP=params.M4))


    def set_M(self, M: float, units='GeV') -> None:
        """
        Set the current value of M for this PBH
        :param M: black hole mass, measured in units specified by 'units'
        :param units: units used to measure the black hole mass
        """
        if units not in self._mass_conversions:
            raise RuntimeError('Standard4D.Schwarzschild: unit "{unit}" not understood in constructor'.format(unit=units))

        units_to_GeV = self._mass_conversions[units]
        self.M = M * units_to_GeV

    @property
    @abstractmethod
    def radius(self) -> float:
        """
        query for the current radius of the black hole, measured in 1/GeV
        """
        pass

    @property
    @abstractmethod
    def reff(self) -> float:
        """
        query for the effective radius of the black hole, measured in 1/GeV
        """
        pass

    @property
    @abstractmethod
    def alpha(self) -> float:
        """
        query for the value of alpha, where r_eff = alpha r_h. alpha is dimensionless
        """
        pass

    @property
    @abstractmethod
    def T_Hawking(self) -> float:
        """
        query for the Hawking temperature T, measured in GeV
        """
        pass

    @property
    @abstractmethod
    def t(self) -> float:
        """
        query for the parameter t, which gives the coefficient of R_h in the relationship
        T_Hawking = 1/(t * R_h). t is dimensionless
        """
        pass

class BaseSpinningBlackHole(BaseBlackHole):
    def __init__(self, params, M: float, units='GeV', strict=True) -> None:
        """
        capture basic details about the PBH model, which we pass to the BaseBlackHole superclass
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param units: units used to measure the black hole mass
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__(params, M, units=units, strict=strict)

    @property
    @abstractmethod
    def J_limit(self) -> float:
        """
        spinning black holes in D <= 5 dimensions usually have a maximum possible angular momentum;
        this method should return the current value for that
        :return:
        """
        pass

    @property
    @abstractmethod
    def set_J(self, J: float=None, J_over_Jmax: float=None):
        """
        set angular momentum for this black hole, either by specifying the spin parameter J directly,
        or by specifying J/Jmax
        :param J:
        :param J_over_Jmax:
        :return:
        """

    @property
    @abstractmethod
    def xi_astar_argument(self) -> float:
        """
        query for current value of the a* parameter needed to evaluate the fitting functions for xi(a*)
        :return:
        """
        pass

    @property
    @abstractmethod
    def J_over_Jmax(self) -> float:
        """
        query for the current value of J/Jmax. This is used in preference to a*, which has an ambiguous
        interpretation - it isn't defined in the same way for Kerr (where J = a* J_max) and Myers-Perry
        (where J = (a*/sqrt(1+a*^2)) J_max)
        :return:
        """
        pass
