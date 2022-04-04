import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import LifetimeKit as lkit

# compute PBH evolution
params = lkit.ModelParameters(1E14)
print(params)
engine = lkit.CosmologyEngine(params)

sample = lkit.PBHInstance(engine, 1E10, accretion_efficiency_F=0.1, collapse_fraction_f=0.5)
standard = sample.lifetimes['standard']
standard.mass_plot('PBH_T1E10_mass.pdf')
