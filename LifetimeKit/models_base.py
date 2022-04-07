import numpy as np
import math
from operator import itemgetter
from functools import partial

from .natural_units import Kelvin
from .particle_data import SM_particle_table

T_threshold_tolerance = 1E-8


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

    def xi(xi0, b, c, mass, dof, T_Hawking):
        return dof * xi0 * math.exp(-b*math.pow(mass/T_Hawking, c))

    for record in particle_table.values():
        if 'b' in record:
            xis_massive.append(partial(xi, record['xi0'], record['b'], record['c'], record['mass'], record['dof']))

        else:
            # massless species have no temperature dependence
            xis_massless += record['dof'] * record['xi0']

    return xis_massless, xis_massive



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
            _T = np.exp(log_T)
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
        '''
        Compute number of relativistic degrees of freedom in the radiation bath at temperature T
        '''

        # find where radiation temperature lies within out threshold list
        index = np.searchsorted(self.SM_thresholds, T, side='left')
        if index >= self.SM_num_thresholds:
            index = self.SM_num_thresholds - 1

        return self.SM_g_values[index]


class BaseStefanBoltzmannLifetimeModel:
    '''
    Shared infrastructure used by all Stefan-Boltzmann-type lifetime models
    '''

    def __init__(self):
        '''
        To speed up computations, we want to cache the number of relativistic degrees of freedom
        available at any given temperature.
        To do that we need to build a list of mass thresholds
        '''

        # build table of mass thresholds associated with Hawking quanta; this is used in the Stefan-Boltzmann
        # approximation
        self.SM_thresholds, self.SM_g_values = build_cumulative_g_table(SM_particle_table)
        self.SM_num_thresholds = len(self.SM_thresholds)


    def g4(self, T_Hawking):
        '''
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into,
        based on Standard Model particles
        '''

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self.SM_thresholds, T_Hawking, side='left')
        if index >= self.SM_num_thresholds:
            index = self.SM_num_thresholds - 1

        return self.SM_g_values[index]


class BaseGreybodyLifetimeModel:
    '''
    Base infrastructure used by all greybody-factor type lifetime models
    '''

    def __init__(self):
        # cache a list of greybody fitting functions
        self.massless_xi, self.massive_xi = build_greybody_xi(SM_particle_table)
