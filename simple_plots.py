import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import LifetimeKit as lkit

# number of points to use on plots
NumPoints = 200

# generate a plot of PBH formation mass vs. formation temperature
def PBHMassPlot(M5, Tlo=1E3, Thi=None, units='gram', collapse_fraction_f=0.5):
    # build a dictionary of unit conversion coefficients
    units_conversion = {'gram': lkit.Gram, 'kilogram': lkit.Kilogram, 'gev': 1.0}

    # check desired units are sensible
    units_lower = units.lower()
    if units_lower not in units_conversion:
        raise RuntimeError('PBHMassPlot: unit "{unit}" not understood in constructor'.format(unit=units))

    if Thi is None:
        Thi = M5

    params = lkit.RS5D.Parameters(M5)
    engine_RS = lkit.RS5D.Model(params)
    engine_4D = lkit.Standard4D.Model(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    unit = units_conversion[units_lower]

    M_values = [collapse_fraction_f * engine_RS.M_Hubble(T=T) / unit for T in T_range]
    M4_values = [collapse_fraction_f * engine_4D.M_Hubble(T=T) / unit for T in T_range]

    plt.figure()
    plt.loglog(T_range, M_values, label='Randall-Sundrum')
    plt.loglog(T_range, M4_values, label='Standard 4D')
    plt.xlabel('Temperature T / GeV')
    plt.ylabel('PBH mass at formation / {units}'.format(units=units))
    plt.legend()
    plt.savefig('formation-mass.pdf')


# generate a plot of PBH formation lengthscale vs. formation temperature
def PBHLengthscalePlot(M5, Tlo=4E3, Thi=None, units='kilometre'):
    # build a dictionary of unit conversion coefficients
    units_conversion = {'metre': lkit.Metre, 'kilometre': lkit.Kilometre, 'mpc': lkit.Mpc}

    # check desired units are sensible
    units_lower = units.lower()
    if units_lower not in units_conversion:
        raise RuntimeError('PBHLengthscalePlot: unit "{unit}" not understood in constructor'.format(unit=units))

    if Thi is None:
        Thi = M5

    params = lkit.RS5D.Parameters(M5)
    engine_RS = lkit.RS5D.Model(params)
    engine_4D = lkit.Standard4D.Model(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    unit = units_conversion[units_lower]

    R_values = [engine_RS.R_Hubble(T=T) / unit for T in T_range]
    R4_values = [engine_4D.R_Hubble(T=T) / unit for T in T_range]

    plt.figure()
    plt.loglog(T_range, R_values, label='Randall-Sundrum')
    plt.loglog(T_range, R4_values, label='Standard 4D')
    plt.xlabel('Temperature T / GeV')
    plt.ylabel('Collapse lengthscale / {units}'.format(units=units))
    plt.legend()
    plt.savefig('formation-lengthscale.pdf')


# generate a plot of PBH formation mass vs. formation lengthscale
def PBHMassScaleRelation(M5, Tlo=4E3, Thi=None, length_units='kilometre', mass_units='gram', collapse_fraction_f=0.5):
    # build a dictionary of unit conversion coefficients
    length_conversion = {'metre': lkit.Metre, 'kilometre': lkit.Kilometre, 'mpc': lkit.Mpc}
    mass_conversion = {'gram': lkit.Gram, 'kilogram': lkit.Kilogram, 'gev': 1.0}

    # check desired units are sensible
    length_units_lower = length_units.lower()
    mass_units_lower = mass_units.lower()

    if length_units_lower not in length_conversion:
        raise RuntimeError('PBHLengthscalePlot: unit "{unit}" not understood in constructor'.format(unit=length_units))

    if mass_units not in mass_conversion:
        raise RuntimeError('PBHMassPlot: unit "{unit}" not understood in constructor'.format(unit=mass_units))

    if Thi is None:
        Thi = M5

    params = lkit.RS5D.Parameters(M5)
    engine_RS = lkit.RS5D.Model(params)
    engine_4D = lkit.Standard4D.Model(params)

    T_range = np.geomspace(Tlo, Thi, num=NumPoints)

    length_unit = length_conversion[length_units_lower]
    mass_unit = mass_conversion[mass_units_lower]

    R_values = [engine_RS.R_Hubble(T=T) / length_unit for T in reversed(T_range)]
    R4_values = [engine_4D.R_Hubble(T=T) / length_unit for T in reversed(T_range)]

    M_values = [collapse_fraction_f * engine_RS.M_Hubble(T=T) / mass_unit for T in reversed(T_range)]
    M4_values = [collapse_fraction_f * engine_4D.M_Hubble(T=T) / mass_unit for T in reversed(T_range)]

    plt.figure()
    plt.loglog(R_values, M_values, label='Randall-Sundrum')
    plt.loglog(R4_values, M4_values, label='Standard 4D')
    plt.xlabel('Lengthscale $\ell$ / {units}'.format(units=length_units))
    plt.ylabel('PBH mass at formation / {units}'.format(units=mass_units))
    plt.legend()
    plt.savefig('PBH-mass-lengthscale.pdf')


# Generate plots
PBHMassPlot(5E12)
PBHLengthscalePlot(5E12)
PBHMassScaleRelation(5E12)
