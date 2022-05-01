import LifetimeKit as lkit

import seaborn as sns
sns.set()

# params = lkit.RS5D.Parameters(3.7584e11)
params = lkit.RS5D.Parameters(1.4652e14)

# models = ['GreybodyRS5D', 'GreybodyStandard4D', 'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
#           'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannStandard4D-noreff',
#           'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannStandard4D-fixedg',
#           'StefanBoltzmannRS5D-fixedN', 'StefanBoltzmannStandard4D-fixedN',
#           'StefanBoltzmannRS5D-noPage', 'StefanBoltzmannStandard4D-noPage']
models = ['GreybodyRS5D', 'GreybodyStandard4D', 'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
          'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannRS5D-fixedN',
          'StefanBoltzmannRS5D-noPage']
# soln = lkit.PBHInstance(params, 9.8115e7, models=models, compute_rates=True)
soln = lkit.PBHInstance(params, 8.4124e12, models=models, compute_rates=True)
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
