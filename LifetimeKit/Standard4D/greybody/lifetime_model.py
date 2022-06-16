import math

from ..cosmology.standard4D import Model, BlackHole
from ...models_base import BaseFriedlanderGreybodyLifetimeModel, StefanBoltzmann4D
from ...greybody_tables.Friedlander import Friedlander_greybody_table_4D, Standard4D_graviton_greybody_table, \
    build_Friedlander_greybody_xi

Const_2Pi = 2.0 * math.pi

class FriedlanderLifetimeModel(BaseFriedlanderGreybodyLifetimeModel):
    """
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional models),
    using Friedlander et al. fitting functions for xi = 8pi f, where f is the Page factor giving
    the integrated Hawking flux
    """
    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        """
        Instantiate a StefanBoltzmann4DLifetimeModel object
        :param engine: a StandardModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_Page_suppression: suppress accretion rate by Page factor of 2.6
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        note that for a greybody lifetime model this applies *only* to accretion, because the correct effective
        radius is baked into the Page f factor
        """

        # invoke superclass constructor
        super().__init__(engine, Model, BlackHole,
                         accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build list of greybody factors associated with RS graviton states
        self._massless_xi, self._massive_xi, self._xi_dict = \
            build_Friedlander_greybody_xi(Friedlander_greybody_table_4D | Standard4D_graviton_greybody_table)

        # Stefan-Boltzmann model is used only for comparison with the Page f function
        self._stefanboltzmann_model = StefanBoltzmann4D(self._params.StefanBoltzmannConstant4D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._logM_end = math.log(self._params.M4)

    def massless_xi(self, PBH):
        return self._massless_xi


    def massive_xi(self, PBH):
        return self._massive_xi

    def xi_species_list(self, PBH):
        return self._xi_dict

    def _dMdt_evaporation(self, T_rad, PBH):
        """
        Compute evaporation rate at a specified temperature of the radiation bath
        and specified black hole properties
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

        # sum over greybody factors to get evaporation rate
        try:
            dM_dt = -(massless_xi + sum([xi(T_Hawking) for xi in massive_xi])) / (Const_2Pi * rh_sq)
        except ZeroDivisionError:
            dM_dt = float("nan")

        return dM_dt

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

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._PBH to its value
        logM = logM_asarray.item()
        PBH_mass = math.exp(logM)
        self._PBH.set_mass(PBH_mass, 'GeV')

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
