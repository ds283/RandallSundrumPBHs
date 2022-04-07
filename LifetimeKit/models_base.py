import numpy as np

from .natural_units import Kelvin
from .particle_data import SM_particle_table, RS_bulk_particle_table

class BaseCosmology:

    def __init__(self, params):
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

        return self.params.RadiationConstant * self.params.gstar * T4


class BaseLifetimeModel:
    '''
    Shared infrastructure used by all lifetime model
    '''

    def g4(self, T_Hawking):
        total_dof = 0

        for data in SM_particle_table.values():
            mass = data['mass']
            dof = data['dof']

            if T_Hawking > mass:
                total_dof += dof

        return total_dof

    def g5_RS(self, T_Hawking):
        total_dof = 0

        for data in RS_bulk_particle_table.values():
            mass = data['mass']
            dof = data['dof']

            if T_Hawking > mass:
                total_dof += dof

        return total_dof
