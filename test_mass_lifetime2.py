import LifetimeKit as lkit

import seaborn as sns
sns.set()

# weird configuration from emission rate calculation
M5_value = 4.40850528823979e+16
T_init = 134960610.811398

# a black hole configuration that evaporates quite early, for testing impact of changing F et al.
# M5_value = 1.3209e16
# T_init = 2.7495e15

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

params = lkit.RS5D.Parameters(M5_value)

engine = lkit.RS5D.Model(params)

# get mass of Hubble volume expressed in GeV
M_Hubble = engine.M_Hubble(T=T_init)

F = 0.372759372
f = 0.395

# compute initial mass in GeV
M_init = f * M_Hubble

model = lkit.RS5D_Friedlander.LifetimeModel(engine, accretion_efficiency_F=F,
                                            use_Page_suppression=True, use_effective_radius=True)
soln = lkit.PBHLifetimeModel(M_init, T_init, model, num_samples=500, compute_rates=True)
# soln.mass_plot('mass_history.pdf')

soln20 = lkit.PBHLifetimeModel(20.0*M_init, T_init, model, num_samples=500, compute_rates=True)
# soln20.mass_plot('mass_history20.pdf')

lifetimes = {'lifetime': soln,
             'lifetime20': soln20}

for label in lifetimes:
    history = lifetimes[label]
    history.dMdt_plot('{label}_rate_history.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann'])
    history.dMdt_relative_plot('{label}_rate_relative.pdf'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation'])
    history.rates_csv('{label}_rate_history.csv'.format(label=label), show_rates=['gluons', 'photons', 'EW_bosons', 'graviton4D', 'graviton5D', 'leptons', 'quarks', 'evaporation', 'stefanboltzmann', 'accretion'])

    print('-- {label}'.format(label=label))
    print('   compute time = {time}'.format(time=history.compute_time))

    lifetime = history.T_lifetime
    if lifetime is not None:
        print('   PBH lifetime = {evapGeV} GeV = {evapKelvin} K'.format(evapGeV=lifetime,
                                                                        evapKelvin=lifetime / lkit.Kelvin))

        shift = history.T_shift if history.T_shift is not None else 0.0
        print('   Lifetime shift = {shiftKelvin} K = {shiftPercent:.2g}%'.format(shiftKelvin=shift/lkit.Kelvin,
                                                                                 shiftPercent=shift/lifetime * 100.0))

        print('   Evaporated = {evapFlag}'.format(evapFlag=history.evaporated))

    m_init = history.M_init
    if m_init is not None:
        print('   PBH initial mass = {initGeV} GeV = {initGram} g'.format(initGeV=m_init,
                                                                          initGram=m_init / lkit.Gram))

    m_final = history.M_final
    if m_final is not None:
        print('   PBH final mass = {finalGeV} GeV = {finalGram} g'.format(finalGeV=m_final,
                                                                          finalGram=m_final / lkit.Gram))
