import LifetimeKit as lkit

from matplotlib import pyplot as plt
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

num_samples = 500

# baseline Friedlander et al. greybody model with J=0
Friedlander_astar0 = lkit.PBHInstance(params, T_init, models=['GreybodyStandard4D'], compute_rates=True,
                                      num_samples=num_samples)

# spinning greybody model with J=0
Spinning_astar0 = lkit.PBHInstance(params, T_init, models=['Kerr'], compute_rates=True,
                                   num_samples=num_samples)

# spinning greybody model with astar=0.9
Spinning_astar0pt9 = lkit.PBHInstance(params, T_init, astar=0.9, models=['Kerr'], compute_rates=True,
                                      num_samples=num_samples)
Spinning_astar0pt9.angular_momentum_plot('Spinning_astar0pt9_J.pdf')

lifetimes = {
    'Friedlander': Friedlander_astar0.lifetimes['GreybodyStandard4D'],
    'Kerr_astar0': Spinning_astar0.lifetimes['Kerr'],
    'Kerr_astar0pt9': Spinning_astar0pt9.lifetimes['Kerr']}

plt.figure()
for label in lifetimes:
    history = lifetimes[label]

    T_rad_values = history.T_sample_points / lkit.Kelvin
    M_values = history.M_sample_points / lkit.Gram

    plt.loglog(T_rad_values, M_values, label='{key}'.format(key=label))

plt.xlabel('Radiation temperature $T$ / Kelvin')
plt.ylabel('PBH mass $M$ / gram')
plt.legend()
plt.savefig('mass_history_compare.pdf')

for label in lifetimes:
    history = lifetimes[label]

    history.dMdt_plot('{label}_dMdt_history.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann'])
    history.dMdt_relative_plot('{label}_dMdt_relative.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation'])
    # history.rates_csv('{label}_rate_history.csv'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann', 'accretion'])
    if hasattr(history, 'J_init'):
        history.dJdt_plot('{label}_dJdt_history.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation'])

    print('-- {label}'.format(label=label))
    print('   compute time = {time}'.format(time=history.compute_time))

    lifetime = history.T_lifetime
    if lifetime is not None:
        print('   PBH lifetime = {evapGeV} GeV = {evapKelvin} K'.format(evapGeV=lifetime,
                                                                        evapKelvin=lifetime / lkit.Kelvin))

        shift = history.T_shift if history.T_shift is not None else 0.0
        print('   Lifetime shift = {shiftKelvin} K = {shiftPercent:.2g}%'.format(shiftKelvin=shift/lkit.Kelvin,
                                                                                 shiftPercent=shift/lifetime * 100.0))

    m_init = history.M_init
    if m_init is not None:
        print('   PBH initial mass = {initGeV} GeV = {initGram} g'.format(initGeV=m_init,
                                                                          initGram=m_init / lkit.Gram))

    m_final = history.M_final
    if m_final is not None:
        print('   PBH final mass = {finalGeV} GeV = {finalGram} g'.format(finalGeV=m_final,
                                                                          finalGram=m_final / lkit.Gram))

    if hasattr(history, 'J_init'):
        J = history.J_init
        if J is not None:
            print('   PBH initial J = {finalJ}'.format(finalJ=J))

    if hasattr(history, 'J_final'):
        J = history.J_final
        if J is not None:
            print('   PBH final J = {finalJ}'.format(finalJ=J))
