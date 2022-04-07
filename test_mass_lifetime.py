import LifetimeKit as lkit

import seaborn as sns
sns.set()

params = lkit.RS5D.Parameters(1.5935e14)

soln = lkit.PBHInstance(params, 6.4584e13)
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
