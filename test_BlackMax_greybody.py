import csv
from math import pi, pow, exp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

TOLERANCE = 1E-8

class BlackMaxSpectrum:

    # labels for the different spins that we will compute
    spins = ['spin0', 'spin1/2', 'spin1']

    # maximum ell mode that is used in the calculation of the greybody factor
    # for available versions of BlackMax, this is 9
    l_max = 9

    # maximum power that is used in the fitting functions
    # for available versions of BlackMax, this is 9
    power_max = 9

    root_path = Path('BlackMax_greybody_database').resolve(strict=True)

    spin_folders = {'spin0': Path('spectrum/rotation/scalar'),
                    'spin1/2': Path('spectrum/rotation/spinor'),
                    'spin1': Path('spectrum/rotation/gauge')}

    spin_stems = {'spin0': 'scalar_{astar:.1f}_{codim}.txt',
                  'spin1/2': 'spinor_{astar:.1f}_{codim}.txt',
                  'spin1': 'vector_{astar:.1f}_{codim}.txt'}

    root_folder = Path('spectrum/rotation')

    # number of sample points to use in x = omega r_h (BlackMax calls this N_spectrum1)
    num_x_samples = 1000

    # maximum x value to use
    max_x = 15.0

    # set up x sample points
    x_samples = np.linspace(start=max_x/num_x_samples, stop=max_x, num=num_x_samples)

    # how many (elll,m) modes are needed for the spin-0, spin-1/2 and spin-1 greybody factors?
    # for the spin-0 mode this is easy; the representation label ell runs over all integer representations
    # from 0 to l_max, and m labels a basis for this representation (of dimension 2ell+1).
    # Hence we have a total of Sum (2ell+1) modes, for elll between 0 and l_max = (1+l_max)^2.
    # For the spin-1 representation the counting is the same, but the (0,0) representation is excluded,
    # so there is one fewer elements.
    # Finally, for spin-1/2 we have half-integer ell and l_max. We have a total of Sum (2ell+1) modes, but now
    # for ell between 0.5 and l_max-0.5. Shifting the summation index to ell' = ekl + 0.5 we get
    # Sum 2ell' for ell' between 1 and l_max = l_max(1+l_max).
    num_modes = {'spin0': (1+l_max)*(1+l_max),
                 'spin1/2': l_max*(1+l_max),
                 'spin1': (1+l_max)*(1+l_max)-1}

    def __init__(self, codimension: int):
        """
        Read in coefficients for greybody fitting functions from the BlackMax greybody database for a specific
        brane codimension, and use them to populate our internal data.
        Once populated, the resulting BlackMaxSpectrum object can be used to evaluate the emission spectrum
        of a black hole
        :param codimension: codimension of brane model (usually 1 for the cases of interest to us)
        """

        # read in 'aa_r' values from the root-level file 'n{codim}.txt';
        # these are (I think) values of the dimensionless Myers-Perry parameter a* = a/r_h at which we sample the
        # greybody factors, corresponding to different values of its angular momentum,
        #
        # J = 2 * M_BH * a / (2+n)
        #
        # where n is the brane codimension = number of extra dimensions in the scenario.
        #
        # The Hawking temperature for such a black hole is
        #
        # T_H = [(n+1) + (n-1)*a*^2] / (4*pi * (1 + a*^2) * r_h)
        #
        # see discussion below (5.16) in the Marco Sampaio thesis (https://inspirehep.net/literature/1306434)
        astar_values_file = self.root_path / self.root_folder / Path('n{codim}.txt'.format(codim=codimension))
        if not (astar_values_file.exists() and astar_values_file.is_file()):
            raise RuntimeError('Could not find astar values file for brane of codim={codim}, '
                               'expected filename was "{fname}"'.format(codim=codimension, fname=astar_values_file))

        self.astar_values = []

        with open(astar_values_file, 'r') as f:
            for line in f:
                try:
                    self.astar_values.append(float(line))
                except ValueError as e:
                    pass

        # record number of aa_r values used in the fitting function
        self.num_astar_values = len(self.astar_values)

        # set up dictionary that will hold the data tables for each table at each value of a*
        self.spectra = {}

        # for each one of these aa_r values, there is a separate file in the BlackMax database that gives the other
        # coefficients used in the fitting function.

        # our job is now to loop through all these values and use them to populate our internal data
        for n, astar in enumerate(self.astar_values):

            # compute BlackMax Omega value
            # this seems to be the angular velocity of the horizon, Omega_H, as defined below Eq. (5.16) of
            # the Marco Sampaio thesis (https://inspirehep.net/literature/1306434), in untis of 1/r_h
            # (note that in the thesis, he uses units of the horizon radius so that r_h = 1; this
            # is spelled out above Eq. (3.53) on p.48).
            # This equation also matches Eq. (51) of the BlackMax paper (http://arxiv.org/abs/0711.3012v4)
            astar_sq = astar * astar
            Omega = astar / (1.0 + astar_sq)

            # compute BlackMax T value
            # this is the black hole temperature, also as defined below Eq. (5.16) of the Marco Sampaio
            # thesis (https://inspirehep.net/literature/1306434), also in units of 1/r_h.
            T = ((codimension+1.0) + (codimension-1.0)*astar_sq) / (4 * pi * (1.0 + astar_sq))

            # populate fitting functions for this value of astar
            table = {}

            # repeat for each spin
            for spin in self.spins:
                file = self.root_path / self.spin_folders[spin] / Path('n{codim}'.format(codim=codimension) /
                        Path(self.spin_stems[spin].format(astar=astar, codim=codimension)))

                # store the ingested mode tables
                modes = {}

                # reserve space for the computed spectrum
                spectrum = np.zeros_like(self.x_samples)

                # reserve space for computed spectrum using low-frequency analytic approximatrion
                lowfreq_spectrum = np.zeros_like(self.x_samples)

                with open(file, 'r') as f:
                    reader = csv.reader(f, delimiter=',')

                    # iterate over all (ell,m) stored in this configuration file
                    for mode in range(0, self.num_modes[spin]):

                        row = next(reader)
                        if len(row) < 3:
                            raise RuntimeError('Too few representation labels in greybody database file "{file}"'.format(file=file))

                        if row[0] != 'l':
                            raise RuntimeError('Expected greybody mode specification to begin with symbol "l"')

                        ell = float(row[1])
                        m = float(row[2])
                        if ell < 0.0-TOLERANCE:
                            raise ValueError('ell value should not be negative (ell={ell}'.format(ell=ell))
                        if not (m >= -ell-TOLERANCE and m <= +ell+TOLERANCE):
                            raise ValueError('representation basis label m should satisfy -ell <= m <= +ell (ell={ell}, m={m})'.format(ell=ell, m=m))

                        row = next(reader)
                        # BlackMax calls the next three values 'locations'.
                        # they seem to record dividing values of x that specify where to transition between different
                        # fitting functions for the greybody spectrum
                        if len(row) < 3:
                            raise RuntimeError('Too few location values in greybody database file "{file}"'.format(file=file))
                        location = [float(row[0]), float(row[1]), float(row[2])]

                        row = next(reader)
                        # BlackMax calls the next two values A_ro
                        if len(row) < 2:
                            raise RuntimeError('Too few A_ro values in greybody database file "{file}"'.format(file=file))
                        A_ro = [float(row[0]), float(row[1])]

                        row = next(reader)
                        # BlackMax calls the next self.power_max values B_ro
                        if len(row) < self.power_max:
                            raise RuntimeError('Too few B_ro values in greybody database file "{file}"'.format(file=file))
                        B_ro = []
                        for i in range(0, self.power_max):
                            B_ro.append(float(row[i]))

                        row = next(reader)
                        # BlackMax calls the next self.power_max values C_ro
                        if len(row) < self.power_max:
                            raise RuntimeError('Too few C_ro values in greybody database file "{file}"'.format(file=file))
                        C_ro = []
                        for i in range(0, self.power_max):
                            C_ro.append(float(row[i]))

                        row = next(reader)
                        # BlackMax calls the next 1 value D_ro
                        if len(row) < 1:
                            raise RuntimeError('Too few D_ro values in greybody database file "{file}"'.format(file=file))
                        D_ro = float(row[0])

                        modes[mode] = {'representation': (ell, m),
                                       'location': location,
                                       'A_ro': A_ro,
                                       'B_ro': B_ro,
                                       'C_ro': C_ro,
                                       'D_ro': D_ro}

                        # compute the contribution of this mode to the spectrum
                        for i in range(0, self.num_x_samples):
                            x = self.x_samples[i]

                            if x < location[0]:
                                spectrum[i] += self._part1(A_ro, x) / x
                            elif x < location[1]:
                                spectrum[i] += self._part23(B_ro, x - location[1]) / x
                            elif x < location[2]:
                                spectrum[i] += self._part23(C_ro, x - location[2]) / x
                            else:
                                spectrum[i] += self._part4(D_ro, x, Omega, T, m, spin) / x

                # BlackMax uses a post-processing step that seems intended to smooth the fitting functions
                # however this changes the normalization of the spectrum in a way that seems to disagree with
                # the analytic estimates of Ida, Oda & Park (http://arxiv.org/abs/hep-th/0212108v4).
                # Maybe they did not care about this because later they renormalize relative to the final
                # point on the spectrum anyway!
                #
                # Here I change the weighting relative to BlackMax so that I preserve the normalization.

                # Currently this is disabled: it doesn't give an obvious improvement relative to
                # sticking with the basic fitting function, and the match to the Ida, Oda & Park analytic
                # approximations is marginally workse

                # prev_spectrum_point = spectrum[0]
                # spectrum[0] = 0.0
                # for i in range(1, self.num_x_samples):
                #     current_spectrum_point = spectrum[i]
                #
                #     # spectrum[i] = (prev_spectrum_point + current_spectrum_point) / 2.0 + prev_spectrum_point
                #     spectrum[i] = (3.0*prev_spectrum_point + current_spectrum_point) / 4.0
                #     prev_spectrum_point = current_spectrum_point

                # Finally, BlackMax renormalizes the power spectrum so that the final x-sample point
                # has amplitude = 1. It's not clear why they do this. Maybe it makes sampling emission events
                # from the power spectrum easier?
                #
                # Currently this is disabled, since it seems to be something we don't want to do

                # final_spectrum_point = spectrum[self.num_x_samples-1]
                # for i in range(0, self.num_x_samples):
                #     spectrum[i] /= final_spectrum_point

                # Test with the analytic approximation of Ida, Oda & Park ((http://arxiv.org/abs/hep-th/0212108v4).
                # NOTE THIS IS FOR codimension=1 ONLY!

                for i in range(0, self.num_x_samples):
                    lowfreq_spectrum[i] = 0.0

                    x = self.x_samples[i]
                    xsq = x * x

                    if spin == 'spin0':
                        for ell, m in [(2,-2), (2,-1), (2,0), (2,1), (2,2), (1,-1), (1,0), (1,1), (0,0)]:
                            Q = (1 + astar_sq) * x - m * astar
                            Qsq = Q * Q

                            if ell == 0:
                                Gamma = 4.0*xsq - 8.0*xsq*xsq
                            elif ell == 1:
                                Gamma = 4.0*Q*x*xsq * (1.0 + Qsq) / 9.0
                            elif ell == 2:
                                Gamma = 16.0*Q*x*xsq*xsq * (1.0 + 5.0*Qsq/4.0 + Qsq*Qsq/4.0) / 2025.0
                            else:
                                raise RuntimeError

                            lowfreq_spectrum[i] += Gamma / (exp(2.0*pi*Q) - 1.0)

                    elif spin == 'spin1/2':
                        for ell, m in [(3.0/2.0, -3.0/2.0), (3.0/2.0, -1.0/2.0), (3.0/2.0, 1.0/2.0), (3.0/2.0, 3.0/2.0), (1.0/2.0, -1.0/2.0), (1.0/2.0, 1.0/2.0)]:
                            Q = (1 + astar_sq) * x - m * astar
                            Qsq = Q * Q

                            if ell > 1.1:
                                Gamma = xsq*xsq * (1.0 + 40.0*Qsq/9.0 + 16.0*Qsq*Qsq/9.0) / 36.0
                            elif ell > 0.1:
                                Gamma = xsq * (1.0 + 4.0*Qsq) - xsq*xsq * (1.0 + 4.0*Qsq)*(1.0 + 4.0*Qsq) / 2.0
                            else:
                                raise RuntimeError

                            lowfreq_spectrum[i] += Gamma / (exp(2.0*pi*Q) + 1.0)

                    elif spin == 'spin1':
                        for ell, m in [(2,-2), (2,-1), (2,0), (2,1), (2,2), (1,-1), (1,0), (1,1)]:
                            Q = (1 + astar_sq) * x - m * astar
                            Qsq = Q * Q

                            if ell == 1:
                                Gamma = 16.0*Q*x*xsq * (1.0 + Qsq) / 9.0
                            elif ell == 2:
                                Gamma = 4.0*Q*x*xsq*xsq * (1.0 + 5.0*Qsq/4.0 + Qsq*Qsq/4.0) / 225.0
                            else:
                                raise RuntimeError

                            lowfreq_spectrum[i] += Gamma / (exp(2.0*pi*Q) - 1.0)

                    lowfreq_spectrum[i] *= x / (2.0 * pi)

                table[spin] = {'modes': modes,
                               'spectrum': spectrum,
                               'lowfreq_spectrum': lowfreq_spectrum}

            self.spectra[n] = {'astar': astar,
                               'table': table}


    # fitting function for low-frequency part of greybody factor
    def _part1(self, A: List[float], x: float) -> float:
        return A[0] * pow(x, A[1])


    # fitting function for mid-frequency part of greybody factor
    def _part23(self, B: List[float], x: float) -> float:
        return sum(b*pow(x, n) for n, b in enumerate(B))


    # fitting function for high-frequency part of greybody factor
    def _part4(self, D: float, x: float, Omega: float, T: float, m: float, spin: str) -> float:
        if spin not in self.spins:
            raise ValueError('Unexpected spin label')

        # BlackMax measures Omega and T in units of 1/r_h, so
        # T_BlackMax = r_h T_true and Omega_BlackMax = r_h Omega_true. Also, x = r_h omega
        # (note Omega = angular velocity of horizon, omega = frequency of emitted Hawking quantum).
        # Therefore in the combination (omega - m Omega_true) / T_true that appears as the statistical weight
        # in the Hawking power spectrum, we can rewrite as (x - m Omega_BlackMax) / T_BlackMax

        # Bose-Einstein factor for bosons
        if spin in ['spin0', 'spin1']:
            return D*x/(exp((x-m*Omega)/T)-1.0)

        # Fermi-Dirac factor for fermions
        if spin in ['spin1/2']:
            return D*x/(exp((x-m*Omega)/T)+1.0)

        raise ValueError('Unhandled value for spin label')


    def write_spectra_to_csv(self, filename: str):
        df_items = []
        spin_map = {'spin0': 0,
                    'spin1/2': 1,
                    'spin1': 2}

        for astar_serial, astar_sample in self.spectra.items():
            spin_table = astar_sample['table']

            for spin in spin_table:
                data = spin_table[spin]

                spectrum = data['spectrum']
                lowfreq_spectrum = data['lowfreq_spectrum']

                for i in range(0, self.num_x_samples):
                    df_items.append({'astar_serial': astar_serial,
                                     'astar': astar_sample['astar'],
                                     'spin': spin_map[spin],
                                     'x_serial': i,
                                     'x_value': self.x_samples[i],
                                     'spectrum_value': spectrum[i],
                                     'lowfreq_spectrum_value': lowfreq_spectrum[i]})

        df = pd.DataFrame(df_items)
        df.index.name = 'index'
        df.to_csv(filename)

sp = BlackMaxSpectrum(1)
