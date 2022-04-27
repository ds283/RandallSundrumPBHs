import math

from ..cosmology.standard4D import Model, BlackHole
from ...models_base import BaseStefanBoltzmannLifetimeModel, StefanBoltzmann4D, build_cumulative_g_table
from ...particle_data import Standard4D_graviton_table

class LifetimeModel(BaseStefanBoltzmannLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 fixed_g4=None):
        '''
        Instantiate a StefanBoltzmann4DLifetimeModel object
        :param engine: a StandardModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''

        # invoke superclass constructor
        super().__init__(engine, Model, BlackHole,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression,
                         extra_4D_states=Standard4D_graviton_table)

        self._stefanboltzmann_model = StefanBoltzmann4D(self._params.StefanBoltzmannConstant4D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._fixed_g4 = fixed_g4

        self._logM_end = math.log(self._params.M4)

    def _rate_evaporation(self, T_rad, M_PBH):
        T_Hawking = M_PBH.T_Hawking

        g4 = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        return self._stefanboltzmann_model.rate(M_PBH, g4=g4)

    def _rate_stefanboltzmann(self, T_rad, M_PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 4D degree of freedom, using all existing settings
        (effective radius, Page suppression, etc.)
        :param T_rad:
        :param M_PBH:
        :return:
        """
        return self._stefanboltzmann_model.rate(M_PBH, g4=1.0)

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
