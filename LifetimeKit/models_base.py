import math
from functools import partial
from operator import itemgetter

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


def build_greybody_xi(particle_table):
    xis_massive = []
    xis_massless = 0.0
    xis = {}

    def xi(xi0, b, c, mass, dof, T_Hawking):
        return dof * xi0 * math.exp(-b*math.pow(mass/T_Hawking, c))

    for label in particle_table:
        record = particle_table[label]
        if 'b' in record:
            f = partial(xi, record['xi0'], record['b'], record['c'], record['mass'],
                        record['dof'] if record['xi-per-dof'] else 1.0)
            xis_massive.append(f)
            xis[label] = f

        else:
            # massless species have no temperature dependence
            q = (record['dof'] if record['xi-per-dof'] else 1.0) * record['xi0']
            xis_massless += q
            xis[label] = q

    return xis_massless, xis_massive, xis



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

    def rate(self, PBH, g4=0.0, g5=1.0):
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

class BaseStefanBoltzmannLifetimeModel(BondiHoyleLyttletonAccretionModel):
    """
    Shared infrastructure used by all Stefan-Boltzmann-type lifetime models
    """

    def __init__(self, engine, Model, BlackHole, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 extra_4D_states=None):
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

        # extra particle states allow us to add extra dofs to the Standard Model ones, if needed.
        # Examples might be the 4D graviton states. We don't always need those, e.g. in a 5D graviton
        # model where the graviton emission needs to be handled differently.
        if extra_4D_states is not None:
            particle_table = particle_table | extra_4D_states

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

    def _rate_accretion(self, T_rad, PBH):
        return self._accretion_model.rate(T_rad, PBH)


class BaseGreybodyLifetimeModel(BondiHoyleLyttletonAccretionModel):
    """
    Base infrastructure used by all greybody-factor type lifetime models
    """

    def __init__(self, engine, Model, BlackHole, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True):
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
            raise RuntimeError('BaseGreybodyLifetimeModel: supplied engine instance is not of expected type')

        self.engine = engine
        self._params = engine.params

        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression

        # create an accretion model instnce
        self._accretion_model = \
            BondiHoyleLyttletonAccretionModel(engine, accretion_efficiency_F, use_effective_radius,
                                              use_Page_suppression)

        # create a PBHModel instance; the value assigned to the mass doesn't matter
        self._PBH = BlackHole(self.engine.params, 1.0, units='gram')

        # cache a list of greybody fitting functions
        self.massless_xi, self.massive_xi, self.xi_dict = build_greybody_xi(SM_particle_table)

    def _sum_xi_list(self, T_rad, PBH, species):
        """
        compute horizon radius in 1/GeV for the species enumerated in 'speies'
        :param T_rad: current temperature of the radiation bath
        :param PBH: current black hole object, used to abstract Hawking temperature calculation
        :param species: list of names of species to include
        :return: emission rate in Mass/Time = [Energy]^2
        """
        rh = PBH.radius
        rh_sq = rh*rh

        # compute Hawking temperature
        T_Hawking = PBH.T_Hawking

        dM_dt = 0.0
        for label in species:
            record = self.xi_dict[label]

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

    def _rate_accretion(self, T_rad, PBH):
        return self._accretion_model.rate(T_rad, PBH)

    def _rate_quarks(self, T_rad, PBH):
        quarks = ['up quark', 'down quark', 'strange quark', 'charm quark', 'bottom quark', 'top quark']
        return self._sum_xi_list(T_rad, PBH, quarks)

    def _rate_leptons(self, T_rad, PBH):
        leptons = ['electron', 'muon', 'tau', 'neutrino']
        return self._sum_xi_list(T_rad, PBH, leptons)

    def _rate_photons(self, T_rad, PBH):
        return self._sum_xi_list(T_rad, PBH, ['photon'])

    def _rate_gluons(self, T_rad, PBH):
        return self._sum_xi_list(T_rad, PBH, ['gluon'])

    def _rate_EW_bosons(self, T_rad, PBH):
        EW_bosons = ['Higgs', 'Z boson', 'W boson']
        return self._sum_xi_list(T_rad, PBH, EW_bosons)
