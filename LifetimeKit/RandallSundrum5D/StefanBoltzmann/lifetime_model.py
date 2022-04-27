import math

import numpy as np

from ..cosmology.RandallSundrum5D import Model, BlackHole
from ...models_base import BaseStefanBoltzmannLifetimeModel, build_cumulative_g_table, StefanBoltzmann5D
from ...particle_data import RS_bulk_particle_table

class LifetimeModel(BaseStefanBoltzmannLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 fixed_g4=None, fixed_g5=None):
        '''
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''

        # invoke superclass constructor
        super().__init__(engine, Model, BlackHole,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build table of dof values for radiation into bulk quanta
        self.bulk_thresholds, self.bulk_g_values = build_cumulative_g_table(RS_bulk_particle_table)
        self.bulk_num_thresholds = len(self.bulk_thresholds)

        self._stefanboltzmann_model = StefanBoltzmann5D(self._params.StefanBoltzmannConstant4D,
                                                        self._params.StefanBoltzmannConstant5D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._fixed_g4 = fixed_g4
        self._fixed_g5 = fixed_g5

        self._logM_end = math.log(self._params.M4)

    def g5(self, T_Hawking):
        '''
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into
        based on bulk species
        '''

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self.bulk_thresholds, T_Hawking, side='left')
        if index >= self.bulk_num_thresholds:
            index = self.bulk_num_thresholds - 1

        return self.bulk_g_values[index]

    def _rate_evaporation(self, T_rad, M_PBH):
        T_Hawking = M_PBH.T_Hawking

        g4 = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        g5 = self._fixed_g5 if self._fixed_g5 is not None else self.g5(T_Hawking)

        return self._stefanboltzmann_model.rate(M_PBH, g4=g4, g5=g5)

    def _rate_stefanboltzmann(self, T_rad, M_PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 4D degree of freedom, using all existing settings
        (effetive radius, Page suppression, etc.)
        :param T_rad:
        :param M_PBH:
        :return:
        """
        return self._stefanboltzmann_model.rate(M_PBH, g4=1.0, g5=0.0)

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._PBH to its value
        logM = logM_asarray.item()
        M_PBH = math.exp(logM)
        self._PBH.set_value(M_PBH, 'GeV')

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        # ACCRETION
        dlogM_dlogT = -self._rate_accretion(T_rad, self._PBH) / (self._PBH.mass * H)

        # EVAPORATION
        dlogM_dlogT += -self._rate_evaporation(T_rad, self._PBH) / (self._PBH.mass * H)

        return dlogM_dlogT
