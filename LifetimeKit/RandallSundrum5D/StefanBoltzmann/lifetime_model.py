import math

import numpy as np

from ..cosmology.RandallSundrum5D import Model, BlackHole
from ...constants import Page_suppression_factor
from ...models_base import BaseStefanBoltzmannLifetimeModel, build_cumulative_g_table
from ...particle_data import RS_bulk_particle_table

Const_PiOver2 = math.pi / 2.0
Const_4Pi = 4.0 * math.pi

class LifetimeModel(BaseStefanBoltzmannLifetimeModel):
    '''
    Evaluate RHS of mass evolution model (assuming a Randall-Sundrum models),
    using a Stefan-Boltzmann limit for the evaporation term
    (i.e. the integrated Hawking flux)
    '''
    def __init__(self, engine: Model, accretion_efficiency_F=0.3,
                 use_effective_radius=True, use_Page_suppression=True,
                 fixed_g4=None, fixed_g5=None):
        '''
        Instantiate a StefanBoltzmann5DLifetimeModel object
        :param engine: a RandallSundrumModel instance to use for calculations
        :param accretion_efficiency_F: efficiency factor for Bondi-Hoyle-Lyttleton accretion
        :param use_effective_radius: whether accretion should use an effective radius rather than the horizon radius
        '''

        # invoke superclass constructor
        super().__init__()

        # build table of dof values for radiation into bulk quanta
        self.bulk_thresholds, self.bulk_g_values = build_cumulative_g_table(RS_bulk_particle_table)
        self.bulk_num_thresholds = len(self.bulk_thresholds)

        if engine is None or not isinstance(engine, Model):
            raise RuntimeError('RandallSundrum5D.StefanBoltzmann.LifetimeModel: supplied engine is not usable')

        self.engine = engine
        self._params = engine.params
        self._SB_4D = self._params.StefanBoltzmannConstant4D
        self._SB_5D = self._params.StefanBoltzmannConstant5D

        # create a PBHModel instance; the value assigned to the mass doesn't matter
        self._M_PBH = BlackHole(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression
        self._fixed_g4 = fixed_g4
        self._fixed_g5 = fixed_g5

        self._logM_end = math.log(self._params.M4)

    def g5(self, T_Hawking):
        '''
        Compute number of relativistic degrees of freedom available for Hawking quanta to radiate into
        based on bulk species
        '''

        # find where T_Hawking lies within out threshold list
        index = np.searchsorted(self.bulk_thresholds, T_Hawking, side='left')
        if index >= self.bulk_num_thresholds:
            index = self.bulk_num_thresholds - 1

        return self.bulk_g_values[index]

    def _rate_accretion(self, T_rad, M_PBH):
        # compute horizon radius in 1/GeV
        rh = M_PBH.radius
        rh_sq = rh*rh

        # compute current energy density rho(T) at this radiation temperature
        rho = self.engine.rho_radiation(T=T_rad)

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = M_PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        dM_dt = math.pi * self._accretion_efficiency_F * alpha_sq * rh_sq * rho

        return dM_dt

    def _rate_evaporation(self, T_rad, M_PBH):
        # compute horizon radius in 1/GeV
        rh = M_PBH.radius
        rh_sq = rh*rh

        # get alpha, the coefficient that turns rh into the effective radius, r_eff = alpha * rh
        alpha = M_PBH.alpha if self._use_effective_radius else 1.0
        alpha_sq = alpha*alpha

        t = M_PBH.t
        t4 = t*t*t*t

        # compute Hawking temperature and effective number of particle species active in the Hawking quanta
        T_Hawking = M_PBH.T_Hawking
        g4_evap = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        g5_evap = self._fixed_g5 if self._fixed_g5 is not None else self.g5(T_Hawking)

        evap_prefactor = Const_4Pi * alpha_sq / (t4 * rh_sq)
        evap_dof = (g4_evap * self._SB_4D + Const_PiOver2 * alpha * g5_evap * self._SB_5D / t)

        dM_dt = -evap_prefactor * evap_dof / (Page_suppression_factor if self._use_Page_suppression else 1.0)

        return dM_dt

    # step the PBH mass, accounting for accretion and evaporation
    def __call__(self, logT_rad, logM_asarray):
        # for some purposes we need the temperature of the radiation bath expressed in GeV
        T_rad = math.exp(logT_rad)

        # also the PBH mass, and reset the PBH model object self._M_PBH to its value
        logM = logM_asarray.item()
        M_PBH = math.exp(logM)
        self._M_PBH.set_value(M_PBH, 'GeV')

        # compute current Hubble rate at this radiation temperature
        H = self.engine.Hubble(T=T_rad)

        # ACCRETION
        dlogM_dlogT = -self._rate_accretion(T_rad, self._M_PBH ) / (self._M_PBH.mass * H)

        # EVAPORATION
        dlogM_dlogT += -self._rate_evaporation(T_rad, self._M_PBH) / (self._M_PBH.mass * H)

        return dlogM_dlogT
