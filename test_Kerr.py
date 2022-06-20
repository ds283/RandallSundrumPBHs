import LifetimeKit as lkit

import seaborn as sns
sns.set()

# a black hole configuration that evaporates quite early, for testing impact of changing F et al.
M5_value = 1.3209e16
T_init = 2.7495e15

# black hole that has not evaporated by present day with M < 1E15 g
# M5_value = 1.9035e10
# T_init = 6.0578e6

# 4D black hole for comparison with above. This one does evaporate by the present day, so what is different?
# M5_value = 1.9035e10
# T_init = 6.5954e8

# black hole in 5D regime for testing effective number of 5D Stefan-Boltzmann degrees of freedom
# M5_value = 3.7584e11
# T_init = 9.8115e7

# canonical test case
# M5_value = 1.4652e14
# T_init = 8.4124e12

params = lkit.RS5D.Parameters(M5=M5_value)

# baseline Friedlander et al. greybody model with J=0
Friedlander_astar0 = lkit.PBHInstance(params, T_init, models=['GreybodyStandard4D'], compute_rates=True)

# Kerr greybody model with J=0
Kerr_astar0 = lkit.PBHInstance(params, T_init, models=['Kerr'], compute_rates=True)

# Kerr greybody model with astar=0.9
Kerr_astar0pt9 = lkit.PBHInstance(params, T_init, astar=0.9, models=['Kerr'], compute_rates=True)

lifetimes = {'Friedlander': Friedlander_astar0.lifetimes['GreybodyStandard4D'],
             'Kerr_astar0': Kerr_astar0.lifetimes['Kerr'],
             'Kerr_astar0pt9': Kerr_astar0pt9.lifetimes['Kerr']}

for label in lifetimes:
    history = lifetimes[label]

    history.dMdt_plot('{label}_rate_history.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann'])
    history.dMdt_relative_plot('{label}_rate_relative.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation'])
    # history.rates_csv('{label}_rate_history.csv'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann', 'accretion'])

    print('-- {label}'.format(label=label))
    print('   compute time = {time}'.format(time=history.compute_time))

    lifetime = history.T_lifetime
    if lifetime is not None:
        print('   PBH lifetime = {evapGeV} GeV = {evapKelvin} K'.format(evapGeV=lifetime,
                                                                        evapKelvin=lifetime / lkit.Kelvin))

        shift = history.T_shift if history.T_shift is not None else 0.0
        print('   Lifetime shift = {shiftKelvin} K = {shiftPercent:.2g}%'.format(shiftKelvin=shift/lkit.Kelvin,
                                                                                 shiftPercent=shift/lifetime * 100.0))

    mass = history.M_final
    if mass is not None:
        print('   PBH final mass = {finalGeV} GeV = {finalGram} g'.format(finalGeV=mass,
                                                                          finalGram=mass / lkit.Gram))
