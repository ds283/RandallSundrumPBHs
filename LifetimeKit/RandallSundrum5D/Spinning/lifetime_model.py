import math

import numpy as np

from ..cosmology.RandallSundrum5D import Model, SpinningBlackHole
from ...greybody_tables.BlackMax import BlackMax_greybody_table_5D, BlackMax_graviton_greybody_table_5D
from ...models_base import BaseSpinningGreybodyLifetimeModel, StefanBoltzmann4D


class LifetimeModel(BaseSpinningGreybodyLifetimeModel):
    """
    Evaluate RHS of the black hole evolution model for a spinning black hole on a Randall-Sundrum brane,
    using BlackMax estimates for the xi values (i.e. Page f and g factors), except for the graviton where
    we do not currently have these

    These emission rates assume all radiated Hawking quanta are massless. That means we have to implement
    manual switch-off of the emission rate when the temperature is smaller than the particle mass.
    """

    # allow type introspection for our associated BlackHole model
    BlackHoleType = Kerr

    def __init__(self, engine: Model, accretion_efficiency_F=0.3, use_Page_suppression=True,
                 use_effective_radius=True):
        """
        Instantiate a lifetime model object using BlackMax emission rates in 5D
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
        self._xi_dict = BlackMax_greybody_table_5D | BlackMax_graviton_greybody_table_5D


        # Stefan-Boltzmann model is used only for comparison with the Page f function
        self._stefanboltzmann_model = StefanBoltzmann5D(self._params.StefanBoltzmannConstant4D,
                                                        self._params.StefanBoltzmannConstant5D,
                                                        use_effective_radius=use_effective_radius,
                                                        use_Page_suppression=use_Page_suppression)


    def xi_species_list(self, PBH):
        return self._xi_dict

    def _dMdt_graviton5D(self, T_rad, PBH):
        """
        Compute emission rate into 5D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dMdt_species(PBH, ['5D graviton'])

    def _dJdt_graviton5D(self, T_rad, PBH):
        """
        Compute angular momentum emission rate into 5D gravitons
        :param T_rad:
        :param PBH:
        :return:
        """
        return self._sum_dJdt_species(PBH, ['5D graviton'])

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
