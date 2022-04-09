import numpy as np

from ...models_base import BaseGreybodyLifetimeModel, build_greybody_xi
from ...particle_data import Standard4D_graviton_table

from ..cosmology.standard4D import Model, BlackHole

Const_Reff_4D = 3.0 * np.sqrt(3.0) / 2.0

Const_4Pi = 4.0 * np.pi
Const_2Pi = 2.0 * np.pi

class LifetimeModel(BaseGreybodyLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a standard 4-dimensional models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_effective_radius=True):
        '''
        Instantiate a StefanBoltzmann4DLifetimeModel object
        :param engine: a StandardModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        for a greybody lifetime model this applies only to accretion
        '''

        # invoke superclass constructor
        super().__init__()

        # build list of greybody factors associated with 4D Einstein graviton states
        massless, massive, dct = build_greybody_xi(Standard4D_graviton_table)

        self.massless_xi += massless
        self.massive_xi += massive
        self.xi_dict = self.xi_dict | dct

        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('Standard4D.greybody.LifetimeModel: supplied engine instance is not usable')

        self.engine = engine
        self._params = engine.params
        self._SB_4D = self._params.StefanBoltzmannConstant4D

        # create a PBHModel instance; the value assigned to the mass doesn't matter
        self._M_PBH = BlackHole(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius

        self._logM_end = np.log(self._params.M4)

    def _rate_accretion(self, T_rad, M_PBH):
        # compute horizon radius in 1/GeV
        rh = self._M_PBH.radius
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = Const_Reff_4D if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        dM_dt = np.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho

        return dM_dt

    def _rate_evaporation(self, T_rad, M_PBH):
        # compute horizon radius in 1/GeV
        rh = self._M_PBH.radius
        rh_sq = rh*rh

        # compute Hawking temperature
        T_Hawking = self._M_PBH.T_Hawking

        # sum over greybody factors to get evaporation rate
        dM_dt = -(self.massless_xi + sum([xi(T_Hawking) for xi in self.massive_xi])) / (Const_2Pi * rh_sq)

        return dM_dt

    def _rate_quarks(self, T_rad, M_PBH):
        quarks = ['up quark', 'down quark', 'strange quark', 'charm quark', 'top quark', 'bottom quark']
        return self._sum_xi_list(T_rad, M_PBH, quarks)

    def _rate_leptons(self, T_rad, M_PBH):
        leptons = ['electron', 'muon', 'tau', 'neutrino']
        return self._sum_xi_list(T_rad, M_PBH, leptons)

    def _rate_photons(self, T_rad, M_PBH):
        return self._sum_xi_list(T_rad, M_PBH, ['photon'])

    def _rate_gluons(self, T_rad, M_PBH):
        return self._sum_xi_list(T_rad, M_PBH, ['gluon'])

    def _rate_EW_bosons(self, T_rad, M_PBH):
        EW_bosons = ['Higgs', 'Z boson', 'W boson']
        return self._sum_xi_list(T_rad, M_PBH, EW_bosons)

    def _rate_graviton4D(self, T_rad, M_PBH):
        return self._sum_xi_list(T_rad, M_PBH, ['4D graviton'])

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = np.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._M_PBH to its value
        logM = logM_asarray.item()
        M_PBH = np.exp(logM)
        self._M_PBH.set_value(M_PBH, 'GeV')

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        # ACCRETION
        dlogM_dlogT = -self._rate_accretion(T_rad, self._M_PBH) / (self._M_PBH.mass * H)

        # EVAPORATION
        dlogM_dlogT += -self._rate_evaporation(T_rad, self._M_PBH) / (self._M_PBH.mass * H)

        return dlogM_dlogT
