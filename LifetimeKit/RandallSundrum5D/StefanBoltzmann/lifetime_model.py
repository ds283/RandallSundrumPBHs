import math

import numpy as np

from ..cosmology.RandallSundrum5D import Model, SpinlessBlackHole
from ...models_base import BaseStefanBoltzmannLifetimeModel, build_cumulative_g_table, StefanBoltzmann5D
from ...particle_data import RS_graviton_particle_table


class LifetimeModel(BaseStefanBoltzmannLifetimeModel):
    """
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    """

    # allow type introspection for our associated BlackHole model
    BlackHoleType = SpinlessBlackHole

    def __init__(self, engine: Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 fixed_g4=None, fixed_g5=None):
        """
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        """

        # invoke superclass constructor
        super().__init__(engine, Model, SpinlessBlackHole,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build table of dof values for radiation into bulk quanta
        self.bulk_thresholds, self.bulk_g_values = build_cumulative_g_table(RS_graviton_particle_table)
        self.bulk_num_thresholds = len(self.bulk_thresholds)

        self._stefanboltzmann_model = StefanBoltzmann5D(self._params.StefanBoltzmannConstant4D,
                                                        self._params.StefanBoltzmannConstant5D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._fixed_g4 = fixed_g4
        self._fixed_g5 = fixed_g5

    def g5(self, T_Hawking):
        """
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into
        based on bulk species
        """

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self.bulk_thresholds, T_Hawking, side='left')
        if index >= self.bulk_num_thresholds:
            index = self.bulk_num_thresholds - 1

        return self.bulk_g_values[index]

    def _dMdt_evaporation(self, T_rad, PBH):
        T_Hawking = PBH.T_Hawking

        g4 = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        g5 = self._fixed_g5 if self._fixed_g5 is not None else self.g5(T_Hawking)

        return self._stefanboltzmann_model.dMdt(PBH, g4=g4, g5=g5)

    def _dMdt_stefanboltzmann(self, T_rad, PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 4D degree of freedom, using all existing settings
        (effective radius, Page suppression, etc.)
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._stefanboltzmann_model.dMdt(PBH, g4=1.0, g5=0.0)

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, state_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._PBH to its value
        logM = state_asarray.item()
        PBH = math.exp(logM)
        self._PBH.set_M(PBH, 'GeV')

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        try:
            # ACCRETION
            dlogM_dlogT = -self._dMdt_accretion(T_rad, self._PBH) / (self._PBH.M * H)

            # EVAPORATION
            dlogM_dlogT += -self._dMdt_evaporation(T_rad, self._PBH) / (self._PBH.M * H)
        except ZeroDivisionError:
            dlogM_dlogT = float("nan")

        return dlogM_dlogT
