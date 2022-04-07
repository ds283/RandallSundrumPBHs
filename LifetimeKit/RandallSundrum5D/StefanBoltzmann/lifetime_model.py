import numpy as np

from ...models_base import BaseLifetimeModel, build_cumulative_g_table
from ...constants import Page_suppression_factor
from ...particle_data import RS_bulk_particle_table

from ..cosmology.RandallSundrum5D import Model, BlackHole

Const_PiOver2 = np.pi / 2.0
Const_4Pi = 4.0 * np.pi

class LifetimeModel(BaseLifetimeModel):
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

        # create a PBHModel instamce; the value assigned to the mass doesn't matter
        self._M_PBH = BlackHole(self.engine.params, 1.0, units='gram')

        self._accretion_efficiency_F = accretion_efficiency_F
        self._use_effective_radius = use_effective_radius
        self._use_Page_suppression = use_Page_suppression
        self._fixed_g4 = fixed_g4
        self._fixed_g5 = fixed_g5

        self._logM_end = np.log(self._params.M4)

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

        t = self._M_PBH.t
        t4 = t*t*t*t

        # compute Hawking temperature and effective number of particle species active in the Hawking quanta
        T_Hawking = self._M_PBH.T_Hawking
        g4_evap = self._fixed_g4 if self._fixed_g4 is not None else self.g4(T_Hawking)
        g5_evap = self._fixed_g5 if self._fixed_g5 is not None else self.g5(T_Hawking)

        evap_prefactor = Const_4Pi * alpha_sq / (self._M_PBH.mass * H * t4 * rh_sq)
        evap_dof = (g4_evap * self._SB_4D + Const_PiOver2 * alpha * g5_evap * self._SB_5D / t)

        dlogM_dlogT += evap_prefactor * evap_dof / (Page_suppression_factor if self._use_Page_suppression else 1.0)

        # x = self._M_PBH.mass / self.engine.M_Hubble(T=T_rad)
        # evap_to_accrete = 4.0 / (self._accretion_efficiency_F * t4 * rh_sq * rh_sq * rho)
        #
        # T_Hawking = self._M_PBH.T_Hawking
        #
        # print('-- integrator called at x = {x:.5g}, M_PBH = {MPBHGeV:.5g} GeV = {MPBHgram:.5g} gram, '
        #       'T = {TGeV:.5g} GeV = {TKelvin:.5g} Kelvin, returning dlogM_dlogT = {out:.5g}, dM/dT = {nolog:.5g}, '
        #       'evap/accrete = {ratio:.5g}, T_Hawking = {THawkGeV:.5g} = {THawkKelvin:.5g} '
        #       'Kelvin'.format(x=x, MPBHGeV=self._M_PBH.mass, MPBHgram=self._M_PBH.mass / Gram,
        #                       TGeV=T_rad, TKelvin=T_rad / Kelvin, out=dlogM_dlogT,
        #                       nolog=self._M_PBH.mass * dlogM_dlogT / T_rad, ratio=evap_to_accrete,
        #                       THawkGeV=T_Hawking, THawkKelvin=T_Hawking/Kelvin))

        return dlogM_dlogT
