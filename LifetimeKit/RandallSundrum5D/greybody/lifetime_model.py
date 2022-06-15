import math

from ..cosmology.RandallSundrum5D import Model, BlackHole
from ...models_base import BaseGreybodyLifetimeModel, build_Friedlander_greybody_xi, StefanBoltzmann5D
from ...greybody_tables.Friedlander import Friedlander_greybody_table_4D, Friedlander_greybody_table_5D, \
    RS_graviton_greybody_table

Const_2Pi = 2.0 * math.pi

class LifetimeModel(BaseGreybodyLifetimeModel):
    """
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    """
    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        """
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        for a greybody lifetime model this applies only to accretion
        """

        # invoke superclass constructor
        super().__init__(engine, Model, BlackHole, accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build list of greybody factors
        self._massless_xi_5D, self._massive_xi_5D, self._xi_dict_5D =\
            build_Friedlander_greybody_xi(Friedlander_greybody_table_5D | RS_graviton_greybody_table)

        self._massless_xi_4D, self._massive_xi_4D, self._xi_dict_4D =\
            build_Friedlander_greybody_xi(Friedlander_greybody_table_4D | RS_graviton_greybody_table)

        self._stefanboltzmann_model = StefanBoltzmann5D(self._params.StefanBoltzmannConstant4D,
                                                        self._params.StefanBoltzmannConstant5D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._logM_end = math.log(self._params.M4)
    def massless_xi(self, PBH):
        if PBH.is_5D:
            return self._massless_xi_5D
        return self._massless_xi_4D

    def massive_xi(self, PBH):
        if PBH.is_5D:
            return self._massive_xi_5D
        return self._massive_xi_4D

    def xi_dict(self, PBH):
        if PBH.is_5D:
            return self._xi_dict_5D
        return self._xi_dict_4D

    def _rate_evaporation(self, T_rad, PBH):
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

    def _rate_graviton5D(self, T_rad, PBH):
        """
        Compute emission rate into 5D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_xi_list(T_rad, PBH, ['5D graviton'])

    def _rate_stefanboltzmann(self, T_rad, PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 4D degree of freedom, using all existing settings
        (effective radius, Page suppression, etc.)
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._stefanboltzmann_model.rate(PBH, g4=1.0, g5=0.0)

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
            dlogM_dlogT = -self._rate_accretion(T_rad, self._PBH) / (self._PBH.M * H)

            # EVAPORATION
            dlogM_dlogT += -self._rate_evaporation(T_rad, self._PBH) / (self._PBH.M * H)
        except ZeroDivisionError:
            dlogM_dlogT = float("nan")

        return dlogM_dlogT
