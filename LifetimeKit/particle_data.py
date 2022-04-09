# spin weights for contribution to energy density; fermions contribute like 7/8 of a particle
boson_weight = 1.0
fermion_weight = 7.0/8.0

# grey-body fitting coefficients, taken from Table II of Friedlander et al., http://arxiv.org/abs/2201.11761v1
xi0_spin0_4D = 0.00187
b_spin0_4D = 0.395
c_spin0_4D = 1.186

xi0_spin0pt5_4D = 0.001013
b_spin0pt5_4D = 0.337
c_spin0pt5_4D = 1.221

xi0_spin1_4D = 0.000423
b_spin1_4D = 0.276
c_spin1_4D = 1.264

xi0_spin2_4D = 0.0000966

xi0_spin2_5D = 0.00972

# table of primary Hawking quanta and masses, adapted from BlackHawk manual (Appendix C, Table 3)
# https://blackhawk.hepforge.org/manuals/blackhawk1.1.pdf
SM_particle_table = \
 {'Higgs': {'mass': 1.2503E2, 'dof': 1, 'spin-weight': boson_weight, 'xi0': xi0_spin0_4D, 'b': b_spin0_4D, 'c': c_spin0_4D},
  'photon': {'mass': 0.0, 'dof': 2, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D},
  'gluon': {'mass': 0.2, 'dof': 16, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D},
  'W boson': {'mass': 8.0403E1, 'dof': 6, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D},
  'Z boson': {'mass': 9.11876E1, 'dof': 3, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D},
  'neutrino': {'mass': 0.0, 'dof': 6, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D},
  'electron': {'mass': 5.109989461E-4, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'muon': {'mass': 1.056583745E-1, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'tau': {'mass': 1.77686, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'up quark': {'mass': 2.2E-3, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'down quark': {'mass': 4.7E-3, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'charm quark': {'mass': 1.27, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'strange quark': {'mass': 9.6E-2, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'top quark': {'mass': 1.7321E2, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D},
  'bottom quark': {'mass': 4.18, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D}}

Standard4D_graviton_table = \
 {'4D graviton': {'mass': 0.0, 'dof': 2.0, 'xi0': xi0_spin2_4D}}

RS_bulk_particle_table = \
 {'5D graviton': {'mass': 0.0, 'dof': 5.0, 'xi0': xi0_spin2_5D}}
