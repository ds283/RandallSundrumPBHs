import math

from ..cosmology.standard4D import Model, Schwarzschild
from ...models_base import BaseStefanBoltzmannLifetimeModel, StefanBoltzmann4D
from ...particle_data import Standard4D_graviton_particle_table


class LifetimeModel(BaseStefanBoltzmannLifetimeModel):
    """
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    """

    # allow type introspection for our associated BlackHole model
    BlackHoleType = Schwarzschild

    def __init__(self, engine: Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 fixed_g4=None):
        """
        Instantiate a StefanBoltzmann4DLifetimeModel object
        :param engine: a StandardModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        """

        # invoke superclass constructor
        super().__init__(engine, Model, Schwarzschild,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression,
                         extra_4D_state_table=Standard4D_graviton_particle_table)

        self._stefanboltzmann_model = StefanBoltzmann4D(self._params.StefanBoltzmannConstant4D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._fixed_g4 = fixed_g4

    def _dMdt_evaporation(self, T_rad, PBH):
        T_Hawking = PBH.T_Hawking

        g4 = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        return self._stefanboltzmann_model.rate(PBH, g4=g4)

    def _dMdt_stefanboltzmann(self, T_rad, PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 4D degree of freedom, using all existing settings
        (effective radius, Page suppression, etc.)
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._stefanboltzmann_model.rate(PBH, g4=1.0)

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
