import math

from ..cosmology.standard4D import Model, Kerr
from ...models_base import BaseSpinningGreybodyLifetimeModel, StefanBoltzmann4D
from ...greybody_tables.Kerr import Kerr_greybody_table_4D, Kerr_graviton_greybody_table_4D


class KerrLifetimeModel(BaseSpinningGreybodyLifetimeModel):
    """
    Evaluate RHS of the black hole evolution model for a spinning Kerr black hole in 4-dimensions,
    using Kerr estimates for the xi values (i.e. Page f and g factors). These were ultimately
    extracted (for d=4, i.e. n=0) from Dong, Kinney & Stojkovic, https://arxiv.org/abs/1511.05642v3.

    These emission rates assume all radiated Hawking quanta are massless and are used in the same way
    as the emission rates for d=4, i.e. n=1, extracted from BlackMax. That means we have to implement
    manual switch-off of the emission rate when the temperature is smaller than the particle mass.

    This model **does not currently  include graviton emission**, although this is mostly
    negligible in 4D anyway.
    """

    # allow type introspection for our associated BlackHole model
    BlackHoleType = Kerr

    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        """
        Instantiate a lifetime model object using Kerr (Dong et al.) greybody emission rates in 4D
        :param engine:
        :param accretion_efficiency_F:
        :param use_Page_suppression:
        :param use_effective_radius:
        """

        # invoke superclass constructor
        super().__init__(engine, Model, Kerr,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build list of emission rates
        self._xi_dict = Kerr_greybody_table_4D | Kerr_graviton_greybody_table_4D

        # Stefan-Boltzmann model is used only for comparison with the greybody emission rates
        self._stefanboltzmann_model = StefanBoltzmann4D(self._params.StefanBoltzmannConstant4D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)


    def xi_species_list(self, PBH):
        return self._xi_dict

    def _dMdt_graviton4D(self, T_rad, PBH):
        """
        Compute mass emission rate into 4D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dMdt_species(PBH, ['4D graviton'])

    def _dJdt_graviton4D(self, T_rad, PBH):
        """
        Compute angular momentum emission rate into 4D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dJdt_species(PBH, ['4D graviton'])

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

    def __call__(self, logT_rad, state_asarray):
        """
        step the PBH mass and angular momentum, accounting for accretion and evaporation
        :param logT_rad:
        :param state_asarray:
        :return:
        """
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # read the PBH mass and angular momentum, then set our self._PBH object to these values
        logM = state_asarray[0]
        logJ = state_asarray[1]
        self._PBH.set_M(math.exp(logM), 'GeV')
        self._PBH.set_J(J=math.exp(logJ))

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        try:
            # ACCRETION
            dlogM_dlogT = -self._dMdt_accretion(T_rad, self._PBH) / (self._PBH.M * H)

            # EVAPORATION
            dlogM_dlogT += -self._dMdt_evaporation(T_rad, self._PBH) / (self._PBH.M * H)
            dlogJ_dlogT = -self._dJdt_evaporation(T_rad, self._PBH) / (self._PBH.J * H)
        except ZeroDivisionError:
            dlogM_dlogT = float("nan")

        return np.asarray([dlogM_dlogT, dlogJ_dlogT])
