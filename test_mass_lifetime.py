import LifetimeKit as lkit

import seaborn as sns
sns.set()

# black hole that has not evaporated by present day with M < 1E15 g
M5_value = 1.9035e10
T_init = 6.0578e6

# 4D black hole for comparison with above. This one does evaporate by the present day, so what is different?
# M5_value = 1.9035e10
# T_init = 6.5954e84

# black hole in 5D regime for testing effective number of 5D Stefan-Boltzmann degrees of freedom
# M5_value = 3.7584e11
# T_init = 9.8115e7

# canonical test case
# M5_value = 1.4652e14
# T_init = 8.4124e12

params = lkit.RS5D.Parameters(M5_value)

# models = ['GreybodyRS5D', 'GreybodyStandard4D', 'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
#           'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannStandard4D-noreff',
#           'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannStandard4D-fixedg',
#           'StefanBoltzmannRS5D-fixedN', 'StefanBoltzmannStandard4D-fixedN',
#           'StefanBoltzmannRS5D-noPage', 'StefanBoltzmannStandard4D-noPage']
models = ['GreybodyRS5D', 'GreybodyStandard4D', 'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
          'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannRS5D-fixedN',
          'StefanBoltzmannRS5D-noPage']
soln = lkit.PBHInstance(params, T_init, models=models, compute_rates=True)
soln.mass_plot('mass_history.pdf')
soln.T_Hawking_plot('T_Hawking_history.pdf', temperature_units='GeV')

for label in soln.lifetimes:
    history = soln.lifetimes[label]
    history.rates_plot('{label}_rate_history.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann'])
    history.rates_relative_plot('{label}_rate_relative.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation'])
    history.rates_csv('{label}_rate_history.csv'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann', 'accretion'])

    print('-- {label}'.format(label=label))
    print('   compute time = {time}'.format(time=history.compute_time))

    lifetime = history.T_lifetime
    print('   PBH lifetime = {evapGeV} GeV = {evapKelvin} K'.format(evapGeV=lifetime,
                                                                    evapKelvin=lifetime / lkit.Kelvin))

    shift = history.T_shift if history.T_shift is not None else 0.0
    print('   Lifetime shift = {shiftKelvin} K = {shiftPercent:.2g}%'.format(shiftKelvin=shift/lkit.Kelvin,
                                                                             shiftPercent=shift/lifetime * 100.0))
