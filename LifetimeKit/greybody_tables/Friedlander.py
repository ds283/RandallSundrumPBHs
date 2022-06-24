import math
from functools import partial

from ..particle_data import _table_merge, SM_particle_base_table

# fitting coefficients for Page f factor, taken from Table II of
# Friedlander et al., http://arxiv.org/abs/2201.11761v1

# SPIN-0 FITTING FUNCTIONS
_xi0_spin0_4D = 0.00187
_b_spin0_4D = 0.395
_c_spin0_4D = 1.186

_xi0_spin0_5D = 0.0167
_b_spin0_5D = 0.333
_c_spin0_5D = 1.236

# SPIN-1/2 FITTING FUNCTIONS
_xi0_spin0pt5_4D = 0.00103
_b_spin0pt5_4D = 0.337
_c_spin0pt5_4D = 1.221

_xi0_spin0pt5_5D = 0.0146
_b_spin0pt5_5D = 0.276
_c_spin0pt5_5D = 1.297

# SPIN-1 FITTING FUNCTIONS
_xi0_spin1_4D = 0.000423
_b_spin1_4D = 0.276
_c_spin1_4D = 1.264

_xi0_spin1_5D = 0.0115
_b_spin1_5D = 0.220
_c_spin1_5D = 1.361

# SPIN-2 FITTING FUNCTIONS
_xi0_spin2_4D = 0.0000966
_xi0_spin2_5D = 0.00972

# 4D Page f factors (in Friedlander et al. notation expressed as xi = 8pi f)

# NOTE Friedlander et al. report xi values *per dof* except for gravitons, for which all dofs are
# already included in the xi0 value
Friedlander_greybody_table_4D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi0': _xi0_spin0_4D, 'b': _b_spin0_4D, 'c': _c_spin0_4D, 'xi-per-dof': True},
  'photon': {'xi0': _xi0_spin1_4D, 'xi-per-dof': True},
  'gluon': {'xi0': _xi0_spin1_4D, 'b': _b_spin1_4D, 'c': _c_spin1_4D, 'xi-per-dof': True},
  'W boson': {'xi0': _xi0_spin1_4D, 'b': _b_spin1_4D, 'c': _c_spin1_4D, 'xi-per-dof': True},
  'Z boson': {'xi0': _xi0_spin1_4D, 'b': _b_spin1_4D, 'c': _c_spin1_4D, 'xi-per-dof': True},
  'neutrino': {'xi0': _xi0_spin0pt5_4D, 'xi-per-dof': True},
  'electron': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'muon': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'tau': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'up quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'down quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'strange quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'charm quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'bottom quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True},
  'top quark': {'xi0': _xi0_spin0pt5_4D, 'b': _b_spin0pt5_4D, 'c': _c_spin0pt5_4D, 'xi-per-dof': True}}
)

# 5D Page f factors (in Friedlander et al. notation expressed as xi = 8pi f)

# NOTE Friedlander et al. report xi values *per dof* except for gravitons, for which all dofs are
# already included in the xi0 value
Friedlander_greybody_table_5D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi0': _xi0_spin0_5D, 'b': _b_spin0_5D, 'c': _c_spin0_5D, 'xi-per-dof': True},
  'photon': {'xi0': _xi0_spin1_5D, 'xi-per-dof': True},
  'gluon': {'xi0': _xi0_spin1_5D, 'b': _b_spin1_5D, 'c': _c_spin1_5D, 'xi-per-dof': True},
  'W boson': {'xi0': _xi0_spin1_5D, 'b': _b_spin1_5D, 'c': _c_spin1_5D, 'xi-per-dof': True},
  'Z boson': {'xi0': _xi0_spin1_5D, 'b': _b_spin1_5D, 'c': _c_spin1_5D, 'xi-per-dof': True},
  'neutrino': {'xi0': _xi0_spin0pt5_5D, 'xi-per-dof': True},
  'electron': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'muon': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'tau': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'up quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'down quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'strange quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'charm quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'bottom quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True},
  'top quark': {'xi0': _xi0_spin0pt5_5D, 'b': _b_spin0pt5_5D, 'c': _c_spin0pt5_5D, 'xi-per-dof': True}}
)

# greybody factors for emission into spin-2 states
Friedlander_graviton_greybody_table_4D = \
 {'4D graviton': {'xi0': _xi0_spin2_4D, 'xi-per-dof': False}}

# greybody factors for emission into *on brane* spin-2 states
Friedlander_graviton_greybody_table_5D = \
 {'5D graviton': {'xi0': _xi0_spin2_5D, 'xi-per-dof': False}}


def build_Friedlander_greybody_xi(xi_table):
    xis_massive = []
    xis_massless = 0.0
    xi_dict = {}

    def xi(xi0, b, c, mass, dof, T_Hawking):
        return dof * xi0 * math.exp(-b*math.pow(mass/T_Hawking, c))

    for label in xi_table:
        record = xi_table[label]
        if 'b' in record:
            f = partial(xi, record['xi0'], record['b'], record['c'], record['mass'],
                        record['dof'] if record['xi-per-dof'] else 1.0)
            xis_massive.append(f)
            xi_dict[label] = f

        else:
            # massless species have no temperature dependence
            # for speed of evaluation during integration, we aggregate these all together
            q = (record['dof'] if record['xi-per-dof'] else 1.0) * record['xi0']
            xis_massless += q
            xi_dict[label] = q

    return xis_massless, xis_massive, xi_dict
