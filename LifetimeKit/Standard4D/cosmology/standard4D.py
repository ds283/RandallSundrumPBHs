import math

from ...models_base import BaseCosmology, BaseBlackHole, BaseSpinningBlackHole
from ...constants import gstar_full_SM

Const_M_H = 4.0 * math.sqrt(3.0) * math.pi
Const_Sqrt_3 = math.sqrt(3.0)
Const_4Pi = 4.0 * math.pi
Const_8Pi = 8.0 * math.pi

Const_Radius_4D = 1.0 / (4.0 * math.pi)
Const_Reff_4D = 3.0 * math.sqrt(3.0) / 2.0


# Analytic lifetime model for 4D Schwarzschild black hole, for evaporation only, neglecting accretion
def Solve_4D_T(Ti, Mi, Mf, gstar, a, g4, sigma4, M4, alpha):
    a_gstar = a * gstar
    a_gstar_sqrt = math.sqrt(a_gstar)

    Ti_sq = Ti*Ti

    alpha_sq = alpha*alpha

    M4_sq = M4*M4

    Mf_over_Mi = Mf/Mi
    DeltaM = Mi * math.pow(1.0 - Mf_over_Mi*Mf_over_Mi*Mf_over_Mi, 1.0/3.0)
    DeltaM_over_M4 = DeltaM / M4
    DeltaM_over_M4_3 = DeltaM_over_M4 * DeltaM_over_M4 * DeltaM_over_M4

    g_factor = g4*sigma4

    A_const = (8.0/3.0*math.sqrt(3.0))*math.pi

    A1 = 1.0/Ti_sq
    A2 = A_const * a_gstar_sqrt * DeltaM_over_M4_3 / (M4_sq * alpha_sq * g_factor)
    A = A1 + A2

    return 1.0/math.sqrt(A)


class Schwarzschild(BaseBlackHole):
    """
    Model for a Schwarzschild black hole. This is specified only by its mass.
    Black hole models can be queried for physical properties, such as radius, Hawking temperature, mass, etc.
    """

    def __init__(self, params, M: float, units='GeV', strict=True):
        """
        Instantiate a Schwarzschild black hole model. This requires only specification of the mass.
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param units: units used to measure the black hole mass
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__(params, M, units=units, strict=strict)

    @property
    def radius(self) -> float:
        """
        query for the Schwarzschild radius of the black hole, measured in 1/GeV
        the formula is R_S = 2MG = M / (4pi M_4^2)
        """
        return Const_Radius_4D * (self.M / self.params.M4) / self.params.M4

    @property
    def reff(self) -> float:
        """
        query for the effective radius, measured in 1/GeV
        formula is 3 sqrt(3) R_h / 2, see e.g. above Eq. (1) of Guedens et al., astro-ph/0208299v2
        or this is a standard calculation using geometrical capture cross-section arguments
        https://physics.stackexchange.com/questions/52315/what-is-the-capture-cross-section-of-a-black-hole-region-for-ultra-relativistic
        """
        return Const_Reff_4D * self.radius

    @property
    def alpha(self) -> float:
        """
        query for the value of alpha, which determines how the effective radius is related to
        the horizon radius
        """
        return Const_Reff_4D

    # query for the 5D Hawking temperature, measured in GeV
    # the relation is T_H = 1/(2pi R_h)
    @property
    def T_Hawking(self) -> float:
        """
        query for the Hawking temperature, measured in GeV
        the relation is T_H = 1/(4pi R_h)
        """
        try:
            return 1.0 / (Const_4Pi * self.radius)
        except ZeroDivisionError:
            pass

        return float("nan")

    @property
    def t(self) -> float:
        """
        query for t, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
        """
        return Const_4Pi

    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True):
        """
        use an analytic lifetime model to determine the final radiation temperature given a current
        radiation temperature and a target final mass
        """
        return Solve_4D_T(Ti_rad, self.M, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                          2.0, self.params.StefanBoltzmannConstant4D, self.params.M4,
                          Const_Reff_4D if use_effective_radius else 1.0)


class Kerr(BaseSpinningBlackHole):
    """
    Model for a Kerr black hole. This is specified by its mass and angular momentum.
    Black holes model can be queried for physical properties, such as radius, Hawking temperature, mass, angular
    momentum, etc.
    """

    def __init__(self, params, M: float, J: float=None, J_over_Jmax: float=None, units='GeV', strict=True):
        """
        Instantiate a Kerr black hole model. This requires specification of the mass M and angular momentum J
        :param params: parameter container
        :param M: black hole mass, in units specified by 'units'
        :param J: (optional) black hole angular momentum, which is dimensionless
        :param J_over_Jmax: (optional) black hole angular momentum, specified as J/Jmax, for 0 <= J/Jmax < 1.
        For Kerr, J/Jmax is the same as the a* parameter. If both are specified, J/Jmax is used in preference.
        If neither is specified, the angular momentum is set to zero.
        :param units: units used to measure the black hole mass
        :param strict: perform stricter validation checks on parameters (defaults to True); may need to be disabled
        to allow construction of BH models with M < M4 that would usually produce a relic
        """
        super().__init__(params, M, units=units, strict=strict)

        # assign current angular momentum value
        # define a 'None' value first, in order to define all instance attributes within __init__()
        self.J = None
        self.set_J(J=J, J_over_Jmax=J_over_Jmax)

    @property
    def J_limit(self) -> float:
        """
        query for the maximum allowed J for the current value of M
        :return:
        """
        M_M4 = self.M / self.params.M4
        M_M4_sq = M_M4 * M_M4
        return M_M4_sq / Const_8Pi

    def set_J(self, J: float=None, J_over_Jmax: float=None) -> None:
        """
        Set the current value of J for this PBH
        """
        J_limit = self.J_limit

        if J_over_Jmax is not None:
            # check supplied value of J/Jmax is valid
            if J_over_Jmax < 0.0:
                raise RuntimeError('standard4D.Kerr: angular momentum parameter J/Jmax=a* should not be negative '
                                   '(requested value was J/Jmax={JJmax})'.format(JJmax=J_over_Jmax))
            if J_over_Jmax > 1.0:
                raise RuntimeError('standard4D.Kerr: angular momentum parameter J/Jmax=a* should be less than unity '
                                   '(requested value was J/Jmax={JJmax})'.format(JJmax=J_over_Jmax))

            # compute J in terms of J/Jmax
            self.J = J_over_Jmax * J_limit

        elif J is not None:
            # check supplied value of J is valid
            if J < 0.0:
                raise RuntimeError('standard4D.Kerr: angular momentum J should not be negative '
                                   '(requested value was J={J})'.format(J=J))

            if J > J_limit:
                raise RuntimeError('standard4D.Kerr: angular momentum J exceeds maximum allowed value Jmax={Jmax} '
                                   '(requested value was J={J})'.format(Jmax=J_limit, J=J))
            self.J = J

        else:
            raise RuntimeError('standard4D.Kerr: neither J nor J/Jmax=a* was specified')

    @property
    def xi_astar_argument(self) -> float:
        """
        query for current value of the a* parameter needed to evaluate the fitting functions for xi(a*)
        """
        return self.J_over_Jmax

    @property
    def J_over_Jmax(self) -> float:
        """
        query for the current value of J/Jmax. This is used in preference to a*, which has an ambiguous
        interpretation - it isn't defined in the same way for Kerr (where J = a* J_max) and Myers-Perry
        (where J = (a*/sqrt(1+a*^2)) J_max)
        """
        return self.J/self.J_limit

    @property
    def radius(self) -> float:
        """
        query for the current radius of the black hole horizon, measured in 1/GeV
        the formula is R_h = (R_s/2) * (1 + sqrt(1-astar^2)) where R_s = 2MG is the Schwarzschild radius
        """
        astar = self.J_over_Jmax
        Rs = Const_Radius_4D * (self.M / self.params.M4) / self.params.M4
        return (Rs / 2.0) * (1.0 + math.sqrt(1.0 - astar*astar))

    @property
    def reff(self) -> float:
        """
        query for the effective radius of the black hole
        TODO: It's not clear the J=0 value is the right answer here - may need to check the literature,
         or just accept that it's a bodge
        """
        return Const_Reff_4D * self.radius

    @property
    def alpha(self) -> float:
        """
        query for the value of alpha, which determines how the effective radius is related to
        the horizon radius
        TODO: It's not clear the J=0 value is the right answer here - may need to check the literature,
         or just accept that it's a bodge
        """
        return Const_Reff_4D

    @property
    def T_Hawking(self) -> float:
        """
        query for the Hawking temperature, measured in GeV
        the formula is T_H = 1/(4pi R_h) * sqrt(1 - astar^2)
        """
        astar = self.J_over_Jmax
        try:
            return 1.0 / (Const_4Pi * self.radius) * math.sqrt(1.0 - astar*astar)
        except ZeroDivisionError:
            pass

        return float("nan")

    @property
    def t(self) -> float:
        """
        query for the t parameter, which gives the coefficient in the relationship T_Hawking = 1/(t * R_h)
        """
        astar = self.J_over_Jmax
        return Const_4Pi / math.sqrt(1.0 - astar*astar)

    def compute_analytic_Trad_final(self, Ti_rad, relic_scale, use_effective_radius=True) -> float:
        """
        use an analytic lifetime model to determine the final radiation temperature given a current
        radiation temperature and a target final mass
        TODO: does not account for angular momentum – hopefully not important!
        """
        return Solve_4D_T(Ti_rad, self.M, relic_scale, gstar_full_SM, self.params.RadiationConstant,
                          2.0, self.params.StefanBoltzmannConstant4D, self.params.M4,
                          Const_Reff_4D if use_effective_radius else 1.0)


class Model(BaseCosmology):
    """
    # The Model class provides methods to compute the Hubble rate, Hubble length, horizon mass, etc.,
    # for the standard 4-dimensional cosmology
    """

    def __init__(self, params, fixed_g=None):
        super().__init__(params, fixed_g)

    def Hubble(self, T=None, log_T=None):
        """
        compute the Hubble rate in GeV at a time corresponding to a temperature supplied in GeV
        """
        rho = self.rho_radiation(T, log_T)

        return 1.0 / (Const_Sqrt_3 * self._params.M4) * math.sqrt(rho)

    def R_Hubble(self, T=None, log_T=None):
        """
        compute the Hubble length in 1/GeV at a time corresponding to a temperature supplied in GeV
        the formula here is R_H = 1/H
        """
        return 1.0 / self.Hubble(T, log_T)

    def M_Hubble(self, T=None, log_T=None):
        """
        compute the mass (in GeV) enclosed within the Hubble length, at a time corresponding to a temperature supplied in GeV
        """
        rho = self.rho_radiation(T, log_T)
        M_H = Const_M_H * self._params.M4 * self._params.M4 * self._params.M4 / math.sqrt(rho)

        return M_H
