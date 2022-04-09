import LifetimeKit as lkit

import seaborn as sns
sns.set()

params = lkit.RS5D.Parameters(1E12)

# models = ['GreybodyRS5D', 'GreybodyStandard4D', 'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
#           'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannStandard4D-noreff',
#           'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannStandard4D-fixedg',
#           'StefanBoltzmannRS5D-fixedN', 'StefanBoltzmannStandard4D-fixedN',
#           'StefanBoltzmannRS5D-noPage', 'StefanBoltzmannStandard4D-noPage']
models = ['GreybodyRS5D', 'StefanBoltzmannRS5D', 'StefanBoltzmannRS5D-noreff',
          'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannRS5D-fixedN', 'StefanBoltzmannRS5D-noPage']
soln = lkit.PBHInstance(params, 1E10, models=models)
soln.mass_plot('mass_history_fail.pdf')

for label in soln.lifetimes:
    history = soln.lifetimes[label]
    print('-- {label}'.format(label=label))
    print('   compute time = {time}'.format(time=history.compute_time))

    lifetime = history.T_lifetime
    print('   PBH lifetime = {evapGeV} GeV = {evapKelvin} K'.format(evapGeV=lifetime,
                                                                    evapKelvin=lifetime / lkit.Kelvin))

    shift = history.T_shift if history.T_shift is not None else 0.0
    print('   Lifetime shift = {shiftKelvin} K = {shiftPercent:.2g}%'.format(shiftKelvin=shift/lkit.Kelvin,
                                                                             shiftPercent=shift/lifetime * 100.0))
