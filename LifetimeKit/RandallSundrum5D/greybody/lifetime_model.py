import numpy as np

from ...models_base import BaseGreybodyLifetimeModel, build_greybody_xi
from ...particle_data import RS_bulk_particle_table

from ..cosmology.RandallSundrum5D import Model, BlackHole

Const_PiOver2 = np.pi / 2.0
Const_2Pi = 2.0 * np.pi
Const_4Pi = 4.0 * np.pi

class LifetimeModel(BaseGreybodyLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_effective_radius=True):
        '''
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius;
        for a greybody lifetime model this applies only to accretion
        '''

        # invoke superclass constructor
        super().__init__()

        # build list of greybody factors associated with RS graviton states
        massless, massive = build_greybody_xi(RS_bulk_particle_table)

        self.massless_xi += massless
        self.massive_xi += massive

        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('RandallSundrum5D.greybody.LifetimeModel: supplied engine is not usable')

        self.engine = engine
        self._params = engine.params
        self._SB_4D = self._params.StefanBoltzmannConstant4D
        self._SB_5D = self._params.StefanBoltzmannConstant5D

        # create a PBHModel instance; the value assigned to the mass doesn't matter
        self._M_PBH = BlackHole(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius

        self._logM_end = np.log(self._params.M4)

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = np.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._M_PBH to its value
        logM = logM_asarray.item()
        M_PBH = np.exp(logM)
        self._M_PBH.set_value(M_PBH, 'GeV')

        # compute horizon radius in 1/GeV
        rh = self._M_PBH.radius
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = self._M_PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        # ACCRETION

        dlogM_dlogT = -np.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho / (self._M_PBH.mass * H)


        # EVAPORATION

        # compute Hawking temperature
        T_Hawking = self._M_PBH.T_Hawking

        # sum over greybody factors to get evaporation rate
        evap = (self.massless_xi + sum([xi(T_Hawking) for xi in self.massive_xi])) / (Const_2Pi * rh_sq)

        dlogM_dlogT += evap / (self._M_PBH.mass * H)

        return dlogM_dlogT
