import LifetimeKit as lkit

params = lkit.ModelParameters(1E12)
engine = lkit.CosmologyEngine(params)

soln = lkit.PBHInstance(engine, 1E10)
soln.mass_plot('mass_history_T1E10.pdf')
