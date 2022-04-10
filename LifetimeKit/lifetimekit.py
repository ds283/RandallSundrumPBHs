import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
import pandas as pd

from .RandallSundrum5D import StefanBoltzmann as RS5D_StefanBoltzmann
from .RandallSundrum5D import cosmology as RS5D
from .RandallSundrum5D import greybody as RS5D_greybody
from .Standard4D import StefanBoltzmann as Standard4D_StefanBoltzmann
from .Standard4D import cosmology as Standard4D
from .Standard4D import greybody as Standard4D_greybody
from .constants import T_CMB
from .constants import gstar_full_SM
from .natural_units import Kelvin, Kilogram, Gram, SolarMass
from .timer import Timer

# number of T-sample points to capture for PBH lifetime mass/temperature relation
NumTSamplePoints = 200

# default models to compute
_DEFAULT_MODEL_SET = ['StefanBoltzmannRS5D', 'GreybodyRS5D', 'StefanBoltzmannStandard4D', 'GreybodyStandard4D']

class LifetimeObserver:
    '''
    LifetimeObserver is a policy object that decides when to store data about the computed
    PBH model (i.e. mass as a function of T), and also checks whether the integration should abort
    because evaporation has proceeded to the point where a relic has formed
    '''

    def __init__(self, engine, sample_grid, mass_grid, x_grid, relic_mass):
        '''
        Instantiate a LifetimeObserver instance.
        The constructor captures model engine instance. _sample_grid should be a numpy 1d array representing
        points where we want to sample the solution M(T), and mass_grid is an (empty) numpy 1d array of the same shape
        into which the answer will be written
        :param engine: RandallSundrumModel instance to use for computations
        :param sample_grid: soln_grid of sample points for independent variable (here log T)
        :param mass_grid: soln_grid of sample points for dependent variable (here M)
        :param x_grid: soln_grid of sample points for dependent variable (here x)
        '''
        if engine is None:
            raise RuntimeError('LifetimeObserver: supplied model instance is None')

        if sample_grid.shape != mass_grid.shape:
            raise RuntimeError('LifetimeObserver: _sample_grid and mass_grid shapes do not match')

        # capture models engine
        self._engine = engine

        # self.terminated is a flag that is set when the integration should terminate because a relic
        # has formed; self.relic_mass records the PBH mass where we declare a relic forms
        self.relic_mass = relic_mass
        self.terminated = False

        # capture reference to sample soln_grid and data soln_grid
        self._sample_grid = sample_grid
        self._mass_grid = mass_grid
        self._x_grid = x_grid

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
    def __call__(self, logT_rad, logM_asarray):
        '''
        Execute an observation step. This should sample the solution if needed, storing the current value in
        self._mass_grid, and advance self.sample_grid_current_index (and update self.next_sample_point)
        :param logT_rad: current value of log T for the radiation bath
        :param logx_asarray: current value of log x, where x is the PBH mass fraction x = M/M_H
        :return:
        '''
        # for some calculations we cannot avoid using the temperature of the radiation bath
        # expressed in GeV
        T_rad = math.exp(logT_rad)

        # extract current value of PBH mass, in GeV
        M_PBH = math.exp(logM_asarray.item())

        # write solution into M-soln_grid if we have passed an observation point
        if self.next_sample_point is not None and logT_rad < self.next_sample_point:
            # compute mass as a fraction of the Hubble volume mass
            x = M_PBH / self._engine.M_Hubble(T=T_rad)

            # store these values
            self._mass_grid[self.sample_grid_current_index] = M_PBH
            self._x_grid[self.sample_grid_current_index] = x

            self.sample_grid_current_index += 1
            if self.sample_grid_current_index < self._sample_grid_length:
                self.next_sample_point = self._sample_grid[self.sample_grid_current_index]
            else:
                self.next_sample_point = None

        # check whether integration should halt because we have decreased the PBH mass below the 4D Planck scale M4.
        # If this happens, we either get a relic, or at least the standard calculation of Hawking radiation is
        # invalidated, so either way we should stop
        if M_PBH < self.relic_mass:
            self.terminated = True
            return -1

        return 0


class PBHLifetimeModel:
    # conversion factors into GeV for mass units we understand
    _mass_conversions = {'gram': Gram, 'kilogram': Kilogram, 'SolarMass': SolarMass, 'GeV': 1.0}

    # conversion factors into GeV for temperature units we understand
    _temperature_conversions = {'Kelvin': Kelvin, 'GeV': 1.0}

    # conversion factors into GeV for time units
    _time_conversions = {'second': 1.0, 'year': 60.0*60.0*24.0*365.0}


    def __init__(self, M_init, T_rad_init, LifetimeModel, num_samples=NumTSamplePoints):
        '''
        Capture initial values
        :param M_init: initial PBH mass, expressed in GeV
        :param T_rad_init: temperature of radiation bath at formation, expressed in GeV
        :param LifetimeModel: model to use for lifetime calculations
        :param num_samples: number of samples to extract
        '''
        # LifetimeModel should include an engine field to which we can refer
        self._engine = LifetimeModel.engine
        self._params = self._engine.params

        self.M_init = M_init
        self.T_rad_init = T_rad_init

        # integration actually proceeds with log(x)
        self.logM_init = math.log(M_init)

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
        # currently Hubble mass M_H, x = M/M_H
        self.M_sample_points = np.zeros_like(self.logT_sample_points)
        self.x_sample_points = np.zeros_like(self.logT_sample_points)

        # set lifetime to default value of None, indicating that we could not compute it; we'll overwrite
        # this value later
        self.T_lifetime = None

        # if we have to use an analytic solution to get all the way down to the relic scale,
        # keep track of how much we needed to shift by
        self.T_shift = None

        # set compute time to None; will be overwritten later
        self.compute_time = None

        # prepare an observer object using these sample points, using a relic scale set at the
        # four-dimensional Planck scale (which is usually what we want)
        self._relic_scale = self._params.M4
        observer = LifetimeObserver(self._engine, self.logT_sample_points, self.M_sample_points, self.x_sample_points,
                                    self._relic_scale)

        # run the integration
        self._integrate(LifetimeModel, observer)

        # get list of methods in LifetimeModel that can be used to produce a rate, and use these to populate
        # our rates list
        self.rates = {}
        for method in dir(LifetimeModel):
            if method.startswith('_rate_'):
                c = getattr(LifetimeModel, method, None)
                if callable(c):
                    data = np.zeros_like(self.T_sample_points)

                    M_PBH = self._engine.BlackHoleType(self._params, Kilogram, 'GeV')

                    for n in range(0, len(self.T_sample_points)):
                        M_PBH.set_value(self.M_sample_points[n])
                        data[n] = c(self.T_sample_points[n], M_PBH)

                self.rates[method.removeprefix('_rate_')] = data


    def _integrate(self, LifetimeModel, Observer):
        '''

        :param LifetimeModel: callable representing RHS of ODE system
        :param Observer: callable representing solution observer (to record solution at specified sample points)
        :return:
        '''
        # set up stepper; need to use on that supports solout, which the SUNDIALS ones don't seem to do
        stepper = ode(LifetimeModel).set_integrator('dopri5', rtol=1E-8, nsteps=5000)
        stepper.set_solout(Observer)

        # set up initial conditions for the PBH and the radiation bath
        # to keep the numerics sensible, we can't run the integration directly in grams; the numbers get too large,
        # making the integrator need a very small stepsize to keep up
        stepper.set_initial_value(self.logM_init, self.logT_rad_init)

        with Timer() as timer:
            # integrate down to the present CMB temperature, or when the observer notices that the PBH
            # mass has decreased below M4

            # with np.errstate(over='raise', divide='raise'):
            #     try:
            #         while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
            #                 and not Observer.terminated:
            #             stepper.integrate(Observer.next_sample_point - 0.001)
            #     except FloatingPointError as e:
            #         print('Floating point error: {msg}'.format(msg=str(e)))
            #         print('  -- at Minit = {Minit}, T_rad = {Tinit}, M5={M5}'.format(Minit=self.M_init_5D, Tinit=self.T_rad_init, M5=LifetimeModel._params.M5))
            #
            #         # leave lifetime as null to indicate that numerical results were unreliable here
            #         return

            while stepper.successful() and Observer.next_sample_point is not None and stepper.t > self.logT_min \
                    and not Observer.terminated:
                stepper.integrate(Observer.next_sample_point - 0.001)

        self.compute_time = timer.interval

        # truncate unused sample points at end of x_sample_points
        index = Observer.sample_grid_current_index
        if index < self.T_sample_points.size:
            self.T_sample_points = np.resize(self.T_sample_points, index)
            self.logT_sample_points = np.resize(self.logT_sample_points, index)
            self.M_sample_points = np.resize(self.M_sample_points, index)
            self.x_sample_points = np.resize(self.x_sample_points, index)

        M = math.exp(stepper.y.item())
        T_rad = math.exp(stepper.t)

        # if the observer terminated the integration, this is because the PBH evaporation proceeded
        # to the point where we produce a relic, so we can record the lifetime and exit
        if stepper.successful():
            self.T_lifetime = T_rad
            return

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
        M_PBH = self._engine.BlackHoleType(self._engine.params, M, 'GeV')

        # get current radiation temperature in GeV
        Ti_rad = math.exp(stepper.t)

        # compute the final relic formation time using an analytic estimation
        self.T_lifetime = \
            M_PBH.compute_analytic_Trad_final(Ti_rad, self._relic_scale,
                                              use_effective_radius=LifetimeModel._use_effective_radius)

        # record the shift due to using the analytic model
        self.T_shift = Ti_rad - self.T_lifetime

    def rates_plot(self, filename, rates=None, mass_units='gram', time_units='year', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        if time_units not in self._time_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=time_units))

        # if no models specified, plot them all
        if rates is None:
            rates = self.rates.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]
        time_units_to_seconds = self._time_conversions[time_units]

        plt.figure()

        T_values = self.T_sample_points / temperature_units_to_GeV
        for label in rates:
            if label in self.rates:
                history = np.abs(self.rates[label] / mass_units_to_GeV / time_units_to_seconds)

                plt.loglog(T_values, history, label='{key}'.format(key=label))

        plt.xlabel('Temperature T / {unit}'.format(unit=temperature_units))
        plt.ylabel('Mass change rate / {massunit}/{tunit}'.format(massunit=mass_units, tunit=temperature_units))
        plt.legend()
        plt.savefig(filename)

    def rates_csv(self, filename, rates=None, mass_units='gram', time_units='year', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        if time_units not in self._time_conversions:
            raise RuntimeError('PBHLifetimeModel.rates_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=time_units))

        # if no models specified, plot them all
        if rates is None:
            rates = self.rates.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]
        time_units_to_seconds = self._time_conversions[time_units]

        T_values = self.T_sample_points / temperature_units_to_GeV
        data = {'T_rad': T_values}

        for label in rates:
            if label in self.rates:
                history = np.abs(self.rates[label] / mass_units_to_GeV / time_units_to_seconds)

                data[label] = history

        df = pd.DataFrame(data)
        df.index.name = 'index'
        df.to_csv(filename)


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
    def __init__(self, params, T_rad_init: float, models=_DEFAULT_MODEL_SET,
                 accretion_efficiency_F=0.5, collapse_fraction_f=0.5, delta=0.0, num_samples=NumTSamplePoints):
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
                model = RS5D_greybody.LifetimeModel(engine_RS, accretion_efficiency_F=accretion_efficiency_F,
                                                    use_effective_radius=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'GreybodyStandard4D':
                model = Standard4D_greybody.LifetimeModel(engine_4D, accretion_efficiency_F=accretion_efficiency_F,
                                                          use_effective_radius=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannRS5D':
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannStandard4D':
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannRS5D-noreff':
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=False, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannStandard4D-noreff':
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=False, use_Page_suppression=True)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannRS5D-fixedg':
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True,
                                                           fixed_g4=gstar_full_SM, fixed_g5=5.0)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannStandard4D-fixedg':
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True,
                                                                 fixed_g4=gstar_full_SM)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannRS5D-fixedN':
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS_fixedN,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=True,
                                                           fixed_g4=gstar_full_SM, fixed_g5=5.0)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannStandard4D-fixedN':
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D_fixedN,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=True,
                                                                 fixed_g4=gstar_full_SM)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannRS5D-noPage':
                model = RS5D_StefanBoltzmann.LifetimeModel(engine_RS,
                                                           accretion_efficiency_F=accretion_efficiency_F,
                                                           use_effective_radius=True, use_Page_suppression=False,
                                                           fixed_g4=gstar_full_SM, fixed_g5=5.0)
                self.lifetimes[label] = PBHLifetimeModel(M_init_5D, T_rad_init, model, num_samples=num_samples)

            elif label == 'StefanBoltzmannStandard4D-noPage':
                model = Standard4D_StefanBoltzmann.LifetimeModel(engine_4D,
                                                                 accretion_efficiency_F=accretion_efficiency_F,
                                                                 use_effective_radius=True, use_Page_suppression=False,
                                                                 fixed_g4=gstar_full_SM)
                self.lifetimes[label] = PBHLifetimeModel(M_init_4D, T_rad_init, model, num_samples=num_samples)

            else:
                raise RuntimeError('LifetimeKit.PBHInstance: unknown model type "{label}"'.format(label=label))


    # produce plot of PBH mass over its lifetime, as a function of temperature T
    def mass_plot(self, filename, models=None, mass_units='gram', temperature_units='Kelvin'):
        # check desired units are sensible
        if mass_units not in self._mass_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot(): unit "{unit}" not understood in '
                               'constructor'.format(unit=mass_units))

        if temperature_units not in self._temperature_conversions:
            raise RuntimeError('PBHLifetimeModel.lifetime_plot: unit "{unit}" not understood in '
                               'constructor'.format(unit=temperature_units))

        # if no models specifies, plot them all
        if models is None:
            models = self.lifetimes.keys()

        mass_units_to_GeV = self._mass_conversions[mass_units]
        temperature_units_to_GeV = self._temperature_conversions[temperature_units]

        plt.figure()

        for label in models:
            if label in self.lifetimes:
                history = self.lifetimes[label]
                T_values = history.T_sample_points / temperature_units_to_GeV
                M_values = history.M_sample_points / mass_units_to_GeV

                plt.loglog(T_values, M_values, label='{key}'.format(key=label))

        plt.xlabel('Temperature T / {unit}'.format(unit=temperature_units))
        plt.ylabel('PBH mass / {unit}'.format(unit=mass_units))
        plt.legend()
        plt.savefig(filename)
