import math

from ..cosmology.standard4D import Model, Schwarzschild
from ...models_base import BaseFriedlanderGreybodyLifetimeModel, StefanBoltzmann4D
from ...greybody_tables.Friedlander import Friedlander_greybody_table_4D, Friedlander_graviton_greybody_table_4D, \
    build_Friedlander_greybody_xi

Const_2Pi = 2.0 * math.pi

class FriedlanderLifetimeModel(BaseFriedlanderGreybodyLifetimeModel):
    """
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional models),
    using Friedlander et al. fitting functions for xi = 8pi f, where f is the Page factor giving
    the integrated Hawking flux
    """

    # allow type introspection for our associated BlackHole model
    BlackHoleType = Schwarzschild

    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        """
        Instantiate a lifetime model object using Friedlander et al. greybody emission rates in 4D
        :param engine: a StandardModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_Page_suppression: suppress accretion rate by Page factor of 2.6
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        note that for a greybody lifetime model this applies *only* to accretion, because the correct effective
        radius is baked into the Page f factor
        """

        # invoke superclass constructor
        super().__init__(engine, Model, Schwarzschild,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build list of emission rates
        self._massless_xi, self._massive_xi, self._xi_dict = \
            build_Friedlander_greybody_xi(Friedlander_greybody_table_4D | Friedlander_graviton_greybody_table_4D)

        # Stefan-Boltzmann model is used only for comparison with the greybody emission rates
        self._stefanboltzmann_model = StefanBoltzmann4D(self._params.StefanBoltzmannConstant4D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

    def massless_xi(self, PBH):
        return self._massless_xi


    def massive_xi(self, PBH):
        return self._massive_xi

    def xi_species_list(self, PBH):
        return self._xi_dict

    def _dMdt_graviton4D(self, T_rad, PBH):
        """
        Compute emission rate into 4D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dMdt_species(PBH, ['4D graviton'])

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
        step the PBH mass, accounting for accretion and evaporation
        :param logT_rad:
        :param state_asarray:
        :return:
        """
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._PBH to its value
        logM = state_asarray.item()
        PBH_mass = math.exp(logM)
        self._PBH.set_M(PBH_mass, 'GeV')

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
