import numpy as np

Const_8Pi = 8.0 * np.pi

# CONVERSION FROM NATURAL UNITS TO SI UNITS

# 4D reduced Planck mass measured in GeV
M4 = 2.435E18

# in units where c = hbar = kBoltzmann = 1, the Planck temperature is
# TP = 1/sqrt(G) = 1.41678416E32 Kelvin
#    = sqrt(8pi) / sqrt(8piG) = sqrt(8pi) * M4
# that gives us 1 Kelvin measured in GeV
Kelvin = np.sqrt(Const_8Pi) * M4 / 1.41678416E32

# in units where c = hbar = 1, the Planck length is
# ellP = sqrt(G) = 1.61625518E-35 m
#      = sqrt(8piG) / sqrt(8pi) = 1 / (M4 * sqrt(8pi))
# that gives us 1 metre measured in 1/GeV
Metre = 1.0 / (M4 * np.sqrt(Const_8Pi) * 1.61625518E-35)
Kilometre = 1000.0 * Metre
Mpc = 3.08567758128E+19 * Kilometre

# in units where c = hbar = 1, the Planck mass is
# MP = 1/sqrt(G) = 2.17643424E-8 kg
#    = sqrt(8pi) / sqrt(8piG) = sqrt(8pi) * M4
# that gives us 1 kg measured in GeV
Kilogram = np.sqrt(Const_8Pi) * M4 / 2.17643424E-8
Gram = Kilogram / 1000.0
SolarMass = 1.98847E30 * Kilogram

# in unis where c = hbar = 1, the Planck time is
# MP = sqrt(G) = 5.39124760E-44 s
#    = sqrt(8piG) / sqrt(8pi) = 1 / (M4 * sqrt(8pi))
# that gives us 1 second measured in GeV
Second = 1.0 / (M4 * np.sqrt(Const_8Pi) * 5.39124760E-44)
Minute = 60 * Second
Hour = 60 * Minute
Day = 24 * Hour
Year = 365 * Day
