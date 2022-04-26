import math

from ..cosmology.RandallSundrum5D import Model, BlackHole
from ...models_base import BaseGreybodyLifetimeModel, build_greybody_xi, StefanBoltzmann5D
from ...particle_data import RS_bulk_particle_table

Const_2Pi = 2.0 * math.pi

class LifetimeModel(BaseGreybodyLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        '''
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        for a greybody lifetime model this applies only to accretion
        '''

        # invoke superclass constructor
        super().__init__(engine, Model, BlackHole, accretion_efficiency_F=accretion_efficiency_F,
                         use_effective_radius=use_effective_radius,
                         use_Page_suppression=use_Page_suppression)

        # build list of greybody factors associated with RS graviton states
        massless, massive, dct = build_greybody_xi(RS_bulk_particle_table)

        self.massless_xi += massless
        self.massive_xi += massive

        # merge dictionary from build_greybody_xi() with generic dictionary
        # built by superclass constructor
        self.xi_dict = self.xi_dict | dct

        self._stefanboltzmann_model = StefanBoltzmann5D(self._params.StefanBoltzmannConstant4D,
                                                        self._params.StefanBoltzmannConstant5D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)

        self._logM_end = math.log(self._params.M4)

    def _rate_evaporation(self, T_rad, M_PBH):
        """
        Compute evaporation rate at a specified temperature of the radiation bath
        and specified black hole properties
        :param T_rad:
        :param M_PBH:
        :return:
        """
        # compute horizon radius in 1/GeV
        rh = M_PBH.radius
        rh_sq = rh*rh

        # compute Hawking temperature
        T_Hawking = M_PBH.T_Hawking

        # sum over greybody factors to get evaporation rate
        dM_dt = -(self.massless_xi + sum([xi(T_Hawking) for xi in self.massive_xi])) / (Const_2Pi * rh_sq)

        return dM_dt

    def _rate_graviton5D(self, T_rad, M_PBH):
        """
        Compute emission rate into 5D gravitons
        :param T_rad:
        :param M_PBH:
        :return:
        """
        return self._sum_xi_list(T_rad, M_PBH, ['5D graviton'])

    def _rate_stefanboltzmann(self, T_rad, M_PBH):
        """
        Convenience rate function to return Stefan-Boltzmann emission rate
        for a single 5D degree of freedom, using all existing settings
        (effetive radius, Page suppression, etc.)
        :param T_rad:
        :param M_PBH:
        :return:
        """
        return self._stefanboltzmann_model.rate(M_PBH, g4=0.0, g5=1.0)

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
