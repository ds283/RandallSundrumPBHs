import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
import pandas as pd

from .RandallSundrum5D import StefanBoltzmann as RS5D_StefanBoltzmann
from .RandallSundrum5D import cosmology as RS5D
from .RandallSundrum5D import Friedlander as RS5D_Friedlander
from .RandallSundrum5D import Spinning as RS5D_Spinning
from .Standard4D import StefanBoltzmann as Standard4D_StefanBoltzmann
from .Standard4D import cosmology as Standard4D
from .Standard4D import Friedlander as Standard4D_Friedlander
from .Standard4D import Kerr as Standard4D_Kerr
from .constants import T_CMB
from .constants import gstar_full_SM
from .natural_units import Kelvin, Kilogram, Gram, SolarMass
from .timer import Timer

# number of T-sample points to capture for PBH lifetime mass/temperature relation
_NumTSamplePoints = 200

# default models to compute
_DEFAULT_MODEL_SET = ['StefanBoltzmannRS5D', 'GreybodyRS5D', 'StefanBoltzmannStandard4D', 'GreybodyStandard4D']

_UNIT_ERROR_MESSAGE = 'PBHLifetimeModel: unit "{unit}" not understood in constructor'
_MISSING_HISTORY_MESSAGE = 'History "{label}" not calculated for this PBH lifetime model'

_ASTAR_TOLERANCE = 1E-10

class LifetimeObserver:
    """
    LifetimeObserver is a policy object that decides when to store data about the computed
    PBH model (i.e. mass as a function of T), and also checks whether the integration should abort
    because evaporation has proceeded to the point where a relic has formed
    """

    def __init__(self, engine, BlackHoleType, sample_grid, M_grid, x_grid, T_Hawking_grid, M_relic, M_init,
                 J_init=None, J_grid=None):
        """
        Instantiate a LifetimeObserver instance.
        The constructor captures model engine instance. _sample_grid should be a numpy 1d array representing
        points where we want to sample the solution M(T), and mass_grid is an (empty) numpy 1d array of the same shape
        into which the answer will be written
        :param engine: cosmology model instance to use for computations
        :param BlackHoleType: type of BlackHole instance
        :param sample_grid: grid of sample points for independent variable, T (in GeV)
        :param M_grid: grid of sample points for mass M (in GeV)
        :param x_grid: grid of sample points for x = M/Horizon mass (dimensionles)
        :param T_Hawking_grid: grid of sample points for Hawking temperature T_Hawking (in GeV)
        :param M_relic: relic mass at which to terminate the calculation (in GeV)
        :param M_init: initial PBH mass (in GeV)
        :param J_init: initial PBH angular momentum (dimensionless)
        :param J_grid: grid of sample points for angular momentum J (dimensionless)
        """
        if engine is None:
            raise RuntimeError('LifetimeObserver: supplied model instance is None')

        if sample_grid.shape != M_grid.shape:
            raise RuntimeError('LifetimeObserver: sample_grid and M_grid shapes do not match')
        if sample_grid.shape != x_grid.shape:
            raise RuntimeError('LifetimeObserver: sample_grid and x_grid shapes do not match')
        if sample_grid.shape != T_Hawking_grid.shape:
            raise RuntimeError('LifetimeObserver: sample_grid and T_Hawking_grid shapes do not match')
        if J_grid is not None and sample_grid.shape != J_grid.shape:
            raise RuntimeError('LifetimeObserver: sample_grid and J_grid shapes do not match')

        # capture model's cosmology engine
        self._engine = engine

        # do we need to track J?
        self._using_J = J_init is not None

        # instantiate BlackHole object corresponding to this engine
        if self._using_J:
            self._PBH = BlackHoleType(self._engine.params, M=M_init, J=J_init, units='GeV')
        else:
            self._PBH = BlackHoleType(self._engine.params, M=M_init, units='GeV')

        # self.terminated is a flag that is set when the integration should terminate because a relic
        # has formed; self.M_relic records the PBH mass where we declare a relic forms
        self.relic_mass = M_relic
        self.terminated = False

        # capture maximum mass achieved during lifetime
        self.M_max = M_init

        # capture maximum angular momentum achieved during lifetime, if used
        if self._using_J:
            self.J_max = J_init

        # capture time of 4D to 5D transition (if one occurs)
        # TODO: how should we handle this when there is rotation, and possibly multiple transition events?
        self.T_transition_4Dto5D = None

        # capture time of 5D to 4D transition (if one occurs)
        # TODO: how should we handle this when there is rotation, and possibly multiple transition events?
        self.T_transition_5Dto4D = None

        # is black hole currently in 5D or 4D regime?
        self._PBH_was_5D = self._PBH.is_5D if hasattr(self._PBH, 'is_5D') else None

        # capture reference to sample grid, mass history grid, x history grid (x=PBH mass/horizon mass),
        # and Hawking temperature history grid
        self._sample_grid = sample_grid
        self._M_grid = M_grid
        self._x_grid = x_grid
        self._T_Hawking_grid = T_Hawking_grid

        if self._using_J:
            if J_grid is None:
                raise RuntimeError('LifetimeObserver: if J is supplied, a J sample grid must be supplied also')
            self._J_grid = J_grid

        self._sample_grid_length = sample_grid.size

        # self.sample_grid_current_index is an externally visible data member that exposes our current
        # position within the same soln_grid
        self.sample_grid_current_index = 0

        # self.next_sample_point is an externally visible data member that exposes the value of the
        # next sample poiint
        if self._sample_grid_length > 0:
            self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
        else:
            self.next_sample_point = None

    # observation step should sample the solution if needed, and check whether the integration should end
    def __call__(self, logT_rad, state_asarray):
        """
        Execute an observation step. This should sample the solution if needed, storing the current value in
        self._M_grid, and advance self.sample_grid_current_index (and update self.next_sample_point)
        :param logT_rad: current value of log T for the radiation bath
        :param logx_asarray: current value of log x, where x is the PBH mass fraction x = M/M_H
        :return:
        """
        # for some calculations we cannot avoid using the temperature of the radiation bath
        # expressed in GeV
        T_rad = math.exp(logT_rad)

        # extract current value of PBH mass, in GeV
        M = math.exp(state_asarray[0])
        self._PBH.set_M(M, 'GeV')

        # extract current value of angular momentum, is used
        if self._using_J:
            J = math.exp(state_asarray[1])
            self._PBH.set_J(J)

        # if current mass is larger than previous maximum, reset our running estimate of the maximum
        if self.M_max is None or M > self.M_max:
            self.M_max = M

        # if current angular momentum is larger than previous maximum, reset our running estimate of the maximum
        if self._using_J and (self.J_max is None or J > self.J_max):
            self.J_max = J

        # detect transitions from 5D to 4D and vice versa, if this is a 5D black hole type
        if self._PBH_was_5D is not None:

            # if black hole was 5D at previous observation step, but is no longer 5D, record this and
            # mark the transition point
            if self._PBH_was_5D and not self._PBH.is_5D:
                self.T_transition_5Dto4D = T_rad
                self._PBH_was_5D = False
            elif not self._PBH_was_5D and self._PBH.is_5D:
                self.T_transition_4Dto5D = T_rad
                self._PBH_was_5D = True


        # write solution into M-soln_grid if we have passed an observation point
        if self.next_sample_point is not None and logT_rad < self.next_sample_point:
            # compute mass as a fraction of the Hubble volume mass
            x = M / self._engine.M_Hubble(T=T_rad)

            # store these values
            self._M_grid[self.sample_grid_current_index] = M
            self._x_grid[self.sample_grid_current_index] = x
            self._T_Hawking_grid[self.sample_grid_current_index] = self._PBH.T_Hawking

            if self._using_J:
                self._J_grid[self.sample_grid_current_index] = J

            self.sample_grid_current_index += 1
            if self.sample_grid_current_index < self._sample_grid_length:
                self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
            else:
                self.next_sample_point = None

        # check whether integration should halt because we have decreased the PBH mass below the 4D Planck scale M4.
        # If this happens, we either get a relic, or at least the standard calculation of Hawking radiation is
        # invalidated, so either way we should stop
        if M < self.relic_mass:
            self.terminated = True
            return -1

        return 0


class PBHLifetimeModel:
    """
    Compute the lifetime for a particular PBH parameter combination (specified M, J, ...)
    using a specified lifetime model.
    Performs the integration, and then acts as a container object for the solution.
    """

    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'SolarMass': SolarMass, 'GeV': 1.0}

    # conversion factors into GeV for temperature units we understand
    _temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}

    # conversion factors into GeV for time units
    _time_conversions = {'second': 1.0, 'year': 60.0*60.0*24.0*365.0}


    def __init__(self, M_init, T_rad_init, LifetimeModel, J_init=None, astar_init=None,
                 num_samples=_NumTSamplePoints, compute_rates=False, verbose=False):
        """
        Capture initial values and then integrate the lifetime.
        :param M_init: initial PBH mass, expressed in GeV
        :param T_rad_init: temperature of radiation bath at formation, expressed in GeV
        :param LifetimeModel: model to use for lifetime calculations
        :param J_init: initial PBH angular momentum (a dimensionless quantity) if being used
        :param num_samples: number of samples to extract
        :param compute_rates: perform a post-processing step to compute emission rates for M and J once the solution
        is known
        :param verbose: enable verbose debugging messages
        """
        # LifetimeModel should include an engine field to which we can refer
        self._engine = LifetimeModel.engine
        self._params = self._engine.params

        # print verbose debugging/information messages
        self._verbose = verbose

        # basic sanity checking
        if M_init <= 0.0:
            raise RuntimeError('PBHLifetimeModel: initial mass should be positive')

        # capture whether this integration includes angular momentum
        self._using_J = J_init is not None or astar_init is not None

        # check we can instantiate a PBH instance of the expected type
        # this will check e.g that all constraint on J are respected.
        # We also want to use this to check whether the black hole is 5D at formation (if there is a concept of that)
        if self._using_J:
            PBH = LifetimeModel.BlackHoleType(self._engine.params, M=M_init, astar=astar_init, J=J_init, units='GeV')

            if PBH.J < 0.0:
                raise RuntimeError('PBHLifetimeModel: intitial angular momentum should be non-negative')

            if hasattr(PBH, 'pseudo_astar'):
                # For the spinning Randall-Sundrum 5D black hole this test is best done with the 'pseudo' astar
                if PBH.pseudo_astar < _ASTAR_TOLERANCE:
                    PBH.set_J(astar=_ASTAR_TOLERANCE)
            elif hasattr(PBH, 'astar'):
                if PBH.astar < _ASTAR_TOLERANCE:
                    PBH.set_J(astar=_ASTAR_TOLERANCE)

            # capture initial value for angular momentum; this allows a conversion from astar to J to be done
            # by the PBH instance class
            self.J_init = PBH.J
        else:
            PBH = LifetimeModel.BlackHoleType(self._engine.params, M=M_init, units='GeV')

        # capture initial values for mass and radiation bath
        self.M_init = PBH.M
        self.T_rad_init = T_rad_init

        # integration actually proceeds with log(M) and log(J)
        self.logM_init = math.log(M_init)
        if self._using_J:
            self.logJ_init = math.log(PBH.J)

        # integration is done in terms of log(x) and log(T), where x = M/M_H(T) is the PBH mass expressed
        # as a fraction of the Hubble mass M_H
        self.logT_rad_init = math.log(T_rad_init)

        # sample soln_grid runs from initial temperature of the radiation bath at formation,
        # down to current CMB temmperature T_CMB
        self.T_min = T_CMB * Kelvin
        self.logT_min = math.log(self.T_min)

        self.T_sample_points = np.geomspace(T_rad_init, self.T_min, num_samples)
        self.logT_sample_points = np.log(self.T_sample_points)

        # reserve space for mass history, expressed as a PBH mass in GeV and as a fraction x of the
        # currently Hubble mass M_H, x = M/M_H. Also, we store the history of the Hawking temperature
        self.M_sample_points = np.zeros_like(self.logT_sample_points)
        self.x_sample_points = np.zeros_like(self.logT_sample_points)
        self.T_Hawking_sample_points = np.zeros_like(self.logT_sample_points)

        if self._using_J:
            # reserve space for angular momentum history
            self.J_sample_points = np.zeros_like(self.logT_sample_points)

        # if PBH model has an 'is_5D' property, capture its value; otherwise, record 'None' to indicate that this
        # PBH type has no concept of 4D/5D transition
        if hasattr(PBH, 'is_5D'):
            self.initially_5D = PBH.is_5D
        else:
            self.initially_5D = None

        # set lifetime to default value of None, indicating that we could not compute it; we'll overwrite
        # this value later if the integration is successful
        self.T_lifetime = None

        # if we have to use an analytic solution to get all the way down to the relic scale,
        # keep track of how much we needed to shift by
        self.T_shift = None

        # track final mass
        self.M_final = None

        # track maximum mass
        self.M_max = None

        if self._using_J:
            # track final angular momentum
            self.J_final = None

            # track maximum angular momentum
            self.J_max = None

        # track time of 4D to 5D transition, if one occurs
        # TODO: how should we handle this when there is rotation, and possibly multiple transition events?
        self.T_transition_4Dto5D = None

        # track time of 5D to 4D transition, if one occurs
        # TODO: how should we handle this when there is rotation, and possibly multiple transition events?
        self.T_transition_5Dto4D = None

        # track whether PBH survives until the present day
        self.evaporated = None

        # track whether runaway accretion occurred
        self.runaway_accretion = None

        # set compute time to None; will be overwritten later
        self.compute_time = None

        # prepare an observer object using these sample points, using a relic scale set at the
        # four-dimensional Planck scale (which is usually what we want)
        self._relic_scale = self._params.M4
        if self._using_J:
            observer = LifetimeObserver(self._engine, LifetimeModel.BlackHoleType,
                                        self.logT_sample_points, self.M_sample_points, self.x_sample_points,
                                        self.T_Hawking_sample_points, self._relic_scale, self.M_init,
                                        J_init=self.J_init, J_grid=self.J_sample_points)
        else:
            observer = LifetimeObserver(self._engine, LifetimeModel.BlackHoleType,
                                        self.logT_sample_points, self.M_sample_points, self.x_sample_points,
                                        self.T_Hawking_sample_points, self._relic_scale, self.M_init)

        # run the integration
        self._integrate(LifetimeModel, observer)

        # allocate dictionary for mass (and angular momentum) emission rates; will be populated by
        # _compute_emission_rates() if enabled
        self.dMdt = {}
        self.dJdt = {}

        if compute_rates:
            self._compute_emission_rates(J_init, LifetimeModel)

    def _compute_emission_rates(self, J_init, LifetimeModel):
        for method in dir(LifetimeModel):
            if method.startswith('_dMdt_'):
                rate_name = method.removeprefix('_dMdt_')

                c = getattr(LifetimeModel, method, None)
                if callable(c):
                    # allocate space for this rate history
                    dMdt = np.zeros_like(self.T_sample_points)

                    # instantiate temporary PBH object; initial data does not matter, because it will be
                    # overwritten
                    if self._using_J:
                        PBH = LifetimeModel.BlackHoleType(self._params, M=Kilogram, J=0.0, units='GeV')
                    else:
                        PBH = LifetimeModel.BlackHoleType(self._params, M=Kilogram, units='GeV')

                    for n in range(0, len(self.T_sample_points)):
                        PBH.set_M(self.M_sample_points[n], 'GeV')
                        if self._using_J:
                            PBH.set_J(self.J_sample_points[n])

                        # rate is the plain emission rate, measured in GeV^2 = mass/time
                        dMdt[n] = c(self.T_sample_points[n], PBH)

                    self.dMdt[rate_name] = dMdt

            elif self._using_J and method.startswith('_dJdt_'):
                rate_name = method.removeprefix('_dJdt_')

                c = getattr(LifetimeModel, method, None)
                if callable(c):
                    # allocate space for this rate history
                    dJdt = np.zeros_like(self.T_sample_points)

                    # instantiate temporary PBH object; initial data does not matter, because it will be
                    # overwritten
                    PBH = LifetimeModel.BlackHoleType(self._params, M=Kilogram, J=0.0, units='GeV')

                    for n in range(0, len(self.T_sample_points)):
                        PBH.set_M(self.M_sample_points[n], 'GeV')
                        PBH.set_J(self.J_sample_points[n])

                        # rate is plain emissionr rate, measured in GeV = dimensionless/time
                        dJdt[n] = c(self.T_sample_points[n], PBH)

                    self.dJdt[rate_name] = dJdt

    def _integrate(self, LifetimeModel, Observer):
        """
        Integrate the lifetime of a particular PBH model with a specified observer object
        :param LifetimeModel: callable representing RHS of ODE system
        :param Observer: callable representing solution observer (to record solution at specified sample points)
        :return:
        """
        # set up stepper; need to use on that supports solout, which the SUNDIALS ones don't seem to do
        stepper = ode(LifetimeModel).set_integrator('dopri5', rtol=1E-8, nsteps=10000)
        stepper.set_solout(Observer)

        # set up initial conditions for the PBH and the radiation bath
        # to keep the numerics sensible, we can't run the integration directly in grams; the numbers get too large,
        # making the integrator need a very small stepsize to keep up. So we use log values instead
        if self._using_J:
            stepper.set_initial_value(np.asarray([self.logM_init, self.logJ_init]), self.logT_rad_init)
        else:
            stepper.set_initial_value(self.logM_init, self.logT_rad_init)

        try:
            with Timer() as timer:
                while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
                        and not Observer.terminated:
                    stepper.integrate(Observer.next_sample_point - 0.001)
        except OverflowError as e:
            # if an overflow error occurred, assume this was due to runaway accretion
            self.runaway_accretion = True
        else:
            self.runaway_accretion = False

        # capture compute time
        self.compute_time = timer.interval

        # truncate unused sample points at end of x_sample_points
        index = Observer.sample_grid_current_index
        if index < self.T_sample_points.size:
            self.T_sample_points = np.resize(self.T_sample_points, index)
            self.logT_sample_points = np.resize(self.logT_sample_points, index)
            self.M_sample_points = np.resize(self.M_sample_points, index)
            self.x_sample_points = np.resize(self.x_sample_points, index)
            self.T_Hawking_sample_points = np.resize(self.T_Hawking_sample_points, index)

            if self._using_J:
                self.J_sample_points = np.resize(self.J_sample_points, index)

        M = math.exp(stepper.y[0])
        if self._using_J:
            J = math.exp(stepper.y[1])

        T_rad = math.exp(stepper.t)

        # extract time of 5D -> 4D transition (if one occurs)
        # this isn't affected by integration issues when we get close to the final state
        # (see below for how to handle the 4D to 5D transition)
        self.T_transition_5Dto4D = Observer.T_transition_5Dto4D

        # extract maximum mass value achieved during lifetime from observer
        self.M_max = Observer.M_max

        # if the observer terminated the integration, this is because the PBH evaporation proceeded
        # to the point where we produce a relic (or because the PBH survives until the present day),
        # so we can record the lifetime and exit
        if stepper.successful():
            # extract lifetime and final mass
            self.T_lifetime = T_rad
            self.M_final = M
            if self._using_J:
                self.J_final = J

            # extract time of 4D -> 5D transition (if one occurs); this is reliable since the integration
            # concluded ok
            # if not transition time is recorded, it will basically be the evaporation point
            self.T_transition_4Dto5D = Observer.T_transition_4Dto5D

            # tag with 'evaporated' flag if evaporation occurred
            if Observer.terminated:
                self.evaporated = True
            else:
                self.evaporated = False

            return

        # THIS SECTION ONLY REACHED IF THERE WAS AN INTEGRATION ERROR

        # if there was an integration failure, this is possibly because of a genuine problem, or possibly
        # because we could not resolve the final stages of the integration - because that is very close
        # to a singularity of the ODE system
        code = stepper.get_return_code()
        if code != -3:
            raise RuntimeError('PBH lifetime calculation failed due to an integration error at '
                               'PBH mass M = {MassGeV:.5g} GeV = {MassGram:.5g} gram, '
                               'radiation temperature = {TGeV:.5g} GeV = {TKelvin:.5g} Kelvin, '
                               'code = {code}'.format(MassGeV=M, MassGram=M/Gram,
                                                      TGeV=T_rad, TKelvin=T_rad/Kelvin,
                                                      code=stepper.get_return_code()))

        # this code corresponds to "step size becomes too small", which we interpret to mean
        # that we're close to the point of evaporation down to a relic

        # create an instance of the appropriate black hole type
        if self._using_J:
            PBH = LifetimeModel.BlackHoleType(self._engine.params, M=M, J=J, units='GeV')
        else:
            PBH = LifetimeModel.BlackHoleType(self._engine.params, M=M, units='GeV')

        # get current radiation temperature in GeV
        Ti_rad = math.exp(stepper.t)

        # compute the final relic formation time using an analytic estimation
        use_reff = LifetimeModel._use_effective_radius
        self.T_lifetime = \
            PBH.compute_analytic_Trad_final(Ti_rad, self._relic_scale, use_effective_radius=use_reff)
        self.M_final = self._relic_scale
        # harder to know what to do about J
        if self._using_J:
            self.J_final = J

        # record the shift due to using the analytic model
        self.T_shift = Ti_rad - self.T_lifetime

        # tag as evaporated
        self.evaporated = True

        # compute time that 4D to 5D transition occurs during evaporation
        # note that self._params.M_transition should exist in this scenario
        if Observer.T_transition_4Dto5D is not None:
            self.T_transition_4Dto5D = Observer.T_transition_4Dto5D
        elif hasattr(PBH, 'is_5D') and not PBH.is_5D:
            self.T_transition_4Dto5D = PBH.compute_analytic_Trad_final(Ti_rad, self._params.M_transition,
                                                                       use_effective_radius=use_reff)


    def _validate_units(self, mass_units=None, time_units=None, temperature_units=None):
        # check desired units are sensible
        if mass_units is not None and mass_units not in self._mass_conversions:
            raise RuntimeError(_UNIT_ERROR_MESSAGE.format(unit=mass_units))

        if time_units is not None and time_units not in self._time_conversions:
            raise RuntimeError(_UNIT_ERROR_MESSAGE.format(unit=time_units))

        if temperature_units is not None and temperature_units not in self._temperature_conversions:
            raise RuntimeError(_UNIT_ERROR_MESSAGE.format(unit=temperature_units))

    def dMdt_plot(self, filename, show_rates=None, mass_units='gram', time_units='year', temperature_units='Kelvin'):
        self._validate_units(mass_units=mass_units, time_units=time_units, temperature_units=temperature_units)

        # if no models specified, plot them all
        if show_rates is None or not isinstance(show_rates, list):
            show_rates = self.dMdt.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]
        time_units_to_seconds = self._time_conversions[time_units]

        plt.figure()

        T_values = self.T_sample_points / temperature_units_to_GeV
        for label in show_rates:
            if label in self.dMdt:
                history = np.abs(self.dMdt[label] / mass_units_to_GeV / time_units_to_seconds)

                plt.loglog(T_values, history, label='{key}'.format(key=label))
            else:
                if self._verbose:
                    print(_MISSING_HISTORY_MESSAGE.format(label=label))

        plt.xlabel('Radiation temperature $T_{{\mathrm{{rad}}}}$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('$|dM/dt|$ / {massunit}/{tunit}'.format(massunit=mass_units, tunit=time_units))
        plt.legend()
        plt.savefig(filename)

    def dMdt_relative_plot(self, filename, show_rates=None, compare_rate='stefanboltzmann', temperature_units='Kelvin'):
        # if no models specified, plot them all
        if show_rates is None or not isinstance(show_rates, list):
            show_rates = self.dMdt.keys()

        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        T_values = self.T_sample_points / temperature_units_to_GeV

        if compare_rate not in self.dMdt:
            raise RuntimeError('Comparison rate label "{label}" not present in computed '
                               'dMdt'.format(label=compare_rate))

        compare_history = self.dMdt[compare_rate]

        for label in show_rates:
            if label in self.dMdt:
                history = self.dMdt[label] / compare_history

                plt.semilogx(T_values, history, label='{key}'.format(key=label))
            else:
                if self._verbose:
                    print(_MISSING_HISTORY_MESSAGE.format(label=label))

        plt.xlabel('Radiation temperature $T_{{\mathrm{{rad}}}}$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('$|dM/dt|$ relative to {label}'.format(label=compare_rate))
        plt.legend()
        plt.savefig(filename)


    def rates_csv(self, filename, show_rates=None, mass_units='gram', time_units='year', temperature_units='Kelvin'):
        self._validate_units(mass_units=mass_units, time_units=time_units, temperature_units=temperature_units)

        # if no models specified, plot them all
        if show_rates is None or not isinstance(show_rates, list):
            show_rates = self.dMdt.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]
        time_units_to_seconds = self._time_conversions[time_units]

        T_values = self.T_sample_points / temperature_units_to_GeV
        data = {'T_rad': T_values}

        for label in show_rates:
            if label in self.dMdt:
                history = self.dMdt[label] / mass_units_to_GeV / time_units_to_seconds

                data[label] = history
            else:
                if self._verbose:
                    print(_MISSING_HISTORY_MESSAGE.format(label=label))

        df = pd.DataFrame(data)
        df.index.name = 'index'
        df.to_csv(filename)

    def dJdt_plot(self, filename, show_rates=None, time_units='year', temperature_units='Kelvin'):
        self._validate_units(time_units=time_units, temperature_units=temperature_units)

        # if no models specified, plot them all
        if show_rates is None or not isinstance(show_rates, list):
            show_rates = self.dJdt.keys()

        temperature_units_to_GeV = self._temperature_conversions[temperature_units]
        time_units_to_seconds = self._time_conversions[time_units]

        plt.figure()

        T_values = self.T_sample_points / temperature_units_to_GeV
        for label in show_rates:
            if label in self.dMdt:
                history = np.abs(self.dJdt[label] / time_units_to_seconds)

                plt.loglog(T_values, history, label='{key}'.format(key=label))
            else:
                if self._verbose:
                    print(_MISSING_HISTORY_MESSAGE.format(label=label))

        plt.xlabel('Radiation temperature $T_{{\mathrm{{rad}}}}$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('$|dJ/dt|$ / {tunit}$^{{-1}}$'.format(tunit=time_units))
        plt.legend()
        plt.savefig(filename)


# class PBHInstance captures details of a PBH that forms at a specified initial temperature
# (which we can map to an initial mass and a lengthscale)
class PBHInstance:
    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'SolarMass': SolarMass, 'GeV': 1.0}

    # conversion factors into GeV for temperature units we understand
    _temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}

    # capture models engine instance and formation temperature of the PBH, measured in GeV
    # T_rad_init: temperature of radiation bath at PBH formation
    # accretion_efficiency: accretion efficiency factor F in Bondi-Hoyle-Lyttleton model
    # collapse_fraction: fraction of Hubble volume that collapses to PBH
    # num_sample_ponts: number of T samples to take
    def __init__(self, params, T_rad_init: float, astar: float=None, J: float=None,
                 models=_DEFAULT_MODEL_SET,
                 accretion_efficiency_F: float=0.5, collapse_fraction_f: float=0.5, delta: float=0.0,
                 num_samples: int=_NumTSamplePoints, compute_rates: bool=False):
        self._params = params

        engine_RS = RS5D.Model(params)
        engine_4D = Standard4D.Model(params)

        engine_RS_fixedN = RS5D.Model(params, fixed_g=gstar_full_SM)
        engine_4D_fixedN = Standard4D.Model(params, fixed_g=gstar_full_SM)

        self.accretion_efficiency_F = accretion_efficiency_F

        # x = f (1+delta) is the fraction of the Hubble volume that initially collapses to form the PBH
        x_init = collapse_fraction_f * (1.0 + delta)

        # get mass of Hubble volume expressed in GeV
        M_Hubble_RS = engine_RS.M_Hubble(T=T_rad_init)
        M_Hubble_4D = engine_4D.M_Hubble(T=T_rad_init)

        # compute initial mass in GeV
        M_init_5D = x_init * M_Hubble_RS
        self.M_init_5D = M_init_5D

        M_init_4D = x_init * M_Hubble_4D
        self.M_init_4D = M_init_4D

        self.lifetimes = {}

        for label in models:
            if label == 'GreybodyRS5D':
                if astar is not None or J is not None:
                    raise RuntimeError('GreybodyRS5D lifetime model does not support angular momentum')
                model = RS5D_Friedlander.LifetimeModel(engine_RS, accretion_efficiency_F=accretion_efficiency_F,
                                                       use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'GreybodyStandard4D':
                if astar is not None or J is not None:
                    raise RuntimeError('GreybodyStandard4D lifetime model does not support angular momentum')
                model = Standard4D_Friedlander.LifetimeModel(engine_4D, accretion_efficiency_F=accretion_efficiency_F,
                                                             use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannRS5D':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannRS5D lifetime model does not support angular momentum')
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannStandard4D':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannStandard4D lifetime model does not support angular momentum')
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannRS5D-noreff':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannRS5D-noreff lifetime model does not support angular momentum')
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=False, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannStandard4D-noreff':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannStandard4D-noreff lifetime model does not support angular momentum')
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=False, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannRS5D-fixedg':
                # 'fixedg' is fixed number of degrees of freedom in Hawking quanta
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannRS5D-fixedg lifetime model does not support angular momentum')
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True,
                                                           fixed_g4=gstar_full_SM, fixed_g5=5.0)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannStandard4D-fixedg':
                # 'fixedg' is fixed number of degrees of freedom in Hawking quanta
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannStandard4D-fixedg lifetime model does not support angular momentum')
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True,
                                                                 fixed_g4=gstar_full_SM)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannRS5D-fixedN':
                # 'fixedN' is fixed number of degrees of freedom in Hawking quanta *and* the thermal bath
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannRS5D-fixedN lifetime model does not support angular momentum')
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS_fixedN,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True,
                                                           fixed_g4=gstar_full_SM, fixed_g5=5.0)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannStandard4D-fixedN':
                # 'fixedN' is fixed number of degrees of freedom in Hawking quanta *and* the thermal bath
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannStandard4D-fixedN lifetime model does not support angular momentum')
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D_fixedN,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True,
                                                                 fixed_g4=gstar_full_SM)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannRS5D-noPage':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannRS5D-noPage lifetime model does not support angular momentum')
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=False)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'StefanBoltzmannStandard4D-noPage':
                if astar is not None or J is not None:
                    raise RuntimeError('StefanBoltzmannStandard4D-noPage lifetime model does not support angular momentum')
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=False)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples,
                                                         compute_rates=compute_rates)

            elif label == 'Kerr':
                if astar is None and J is None:
                    print('PBHInstance: initial angular momentum was not supplied for Kerr model; defaulting to astar = 0.0')
                    astar = 0.0
                model = Standard4D_Kerr.LifetimeModel(engine_4D,
                                                      accretion_efficiency_F=accretion_efficiency_F,
                                                      use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model,
                                                         astar_init=astar, J_init=J,
                                                         num_samples=num_samples, compute_rates=compute_rates)

            elif label == 'SpinningRS5D':
                if astar is None and J is None:
                    print('PBHInstance: initial angular momentum was not supplied for RS5D spinning model; defaulting to astar = 0.0')
                    astar = 0.0
                model = RS5D_Spinning.LifetimeModel(engine_RS,
                                                    accretion_efficiency_F=accretion_efficiency_F,
                                                    use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model,
                                                         astar_init=astar, J_init=J,
                                                         num_samples=num_samples, compute_rates=compute_rates)

            else:
                raise RuntimeError('LifetimeKit.PBHInstance: unknown model type "{label}"'.format(label=label))


    def mass_plot(self, filename, models=None, mass_units='gram', temperature_units='Kelvin'):
        """
        produce plot of PBH mass over its lifetime, as a function of the radiation temperature T
        :param filename:
        :param models:
        :param mass_units:
        :param temperature_units:
        :return:
        """
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        # if no models specified, plot them all
        if models is None or not isinstance(models, list):
            models = self.lifetimes.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        for label in models:
            if label in self.lifetimes:
                history = self.lifetimes[label]
                Trad_values = history.T_sample_points / temperature_units_to_GeV
                M_values = history.M_sample_points / mass_units_to_GeV

                plt.loglog(Trad_values, M_values, label='{key}'.format(key=label))

        plt.xlabel('Radiation temperature $T$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH mass $M_{{\mathrm{{PBH}}}}$ / {unit}'.format(unit=mass_units))
        plt.legend()
        plt.savefig(filename)


    def angular_momentum_plot(self, filename, models=None, temperature_units='Kelvin'):
        """
        produce plot of PBH angular momentum over its lifetime, as a function of the radiation temperature T
        :param filename:
        :param models:
        :return:
        """
        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        if models is None or not isinstance(models, list):
            models = self.lifetimes.keys()

        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        for label in models:
            if label in self.lifetimes:
                history = self.lifetimes[label]
                Trad_values = history.T_sample_points / temperature_units_to_GeV
                J_values = history.J_sample_points

                plt.loglog(Trad_values, J_values, label='{key}'.format(key=label))

        plt.xlabel('Radiation temperature $T$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH angular momentum $J_{{\mathrm{{PBH}}}}$')
        plt.legend()
        plt.savefig(filename)

    def T_Hawking_plot(self, filename, models=None, temperature_units='Kelvin'):
        """
        produce plot of PBH Hawking temperature over its lifetime, as a function of the radiation temperature T
        :param filename:
        :param models:
        :param temperature_units:
        :return:
        """
        # check desired units are sensible
        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        # if no models specified, plot them all
        if models is None or not isinstance(models, list):
            models = self.lifetimes.keys()

        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        for label in models:
            if label in self.lifetimes:
                history = self.lifetimes[label]
                Trad_values = history.T_sample_points / temperature_units_to_GeV
                TH_values = history.T_Hawking_sample_points / temperature_units_to_GeV

                plt.loglog(Trad_values, TH_values, label='{key}'.format(key=label))

        plt.xlabel('Radiation temperature $T$ / {unit}'.format(unit=temperature_units))
        plt.ylabel('Hawking temperature $T_{{\mathrm{{H}}}}$ / {unit}'.format(unit=temperature_units))
        plt.legend()
        plt.savefig(filename)
