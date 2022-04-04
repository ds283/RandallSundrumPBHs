import LifetimeKit as lkit

import seaborn as sns
sns.set()

params = lkit.RandallSundrumParameters(1.5935e14)
engine = lkit.CosmologyEngine(params)

soln = lkit.PBHInstance(engine, 6.4584e13)
soln.mass_plot('mass_history_fail.pdf')

SB_4D = soln.lifetimes['StefanBoltzmann4D']
SB_5D = soln.lifetimes['StefanBoltzmann5D']

print('4D lifetime = {Life4D:.5g} Kelvin, 5D lifetime = {Life5D:.5g} Kelvin'.format(Life4D=SB_4D.T_lifetime/lkit.Kelvin,
                                                                                    Life5D=SB_5D.T_lifetime/lkit.Kelvin))
print('4D shift = {Shift4D:.5g} Kelvin, 5D shift = {Shift5D:.5g} Kelvin'.format(Shift4D=SB_4D.T_shift/lkit.Kelvin if SB_4D.T_shift is not None else 0.0,
                                                                                Shift5D=SB_5D.T_shift/lkit.Kelvin if SB_5D.T_shift is not None else 0.0))
