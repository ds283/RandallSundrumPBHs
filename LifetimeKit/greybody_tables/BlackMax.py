import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from ..particle_data import _table_merge, SM_particle_base_table
from .Kerr import xi_dMdt_spin2_spline, xi_dJdt_spin2_spline
from .Friedlander import xi0_spin2_5D, xi0_spin2_4D

# greybody factors extracted from the BlackMax greybody database
astar = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

BlackMax_maximum_astar = 1.5

xi_dMdt_spin0 = np.asarray([0.016733779047978143,
                            0.0168586918073312,
                            0.017274178095103735,
                            0.01806339600973751,
                            0.01981392846868953,
                            0.022271978743540952,
                            0.026141972205832933,
                            0.03157017778111371,
                            0.03904235560627453,
                            0.048603329916217325,
                            0.06026203454156726,
                            0.07455959398972482,
                            0.0900823424713518,
                            0.10529176397792649,
                            0.12021255675192243,
                            0.13413238116000648])

xi_dMdt_spin0pt5 = np.asarray([0.014582027776027216,
                               0.015015116912536744,
                               0.016334158136048806,
                               0.01857853603385632,
                               0.021837790811581017,
                               0.026310227103836913,
                               0.03229437416908495,
                               0.04014778568695689,
                               0.050309153725315456,
                               0.06332192611567004,
                               0.07939667978259118,
                               0.09767208241278501,
                               0.11780112821755317,
                               0.1411502275780105,
                               0.16209417527810932,
                               0.18380421650155893])

xi_dMdt_spin1 = np.asarray([0.011486475223775734,
                            0.012494022921475213,
                            0.015616415782925329,
                            0.021154682342241344,
                            0.02959891475690643,
                            0.0416639735070752,
                            0.05812621547232826,
                            0.0807444727511764,
                            0.11068249171036813,
                            0.14934064970808883,
                            0.19847217791624122,
                            0.2584775713178245,
                            0.33049028705893563,
                            0.4161534431589193,
                            0.5095149046182521,
                            0.6166645966621302])

xi_dJdt_spin0 = np.asarray([4.2250377141904996e-22,
                            0.0029893888632798312,
                            0.006346155977418833,
                            0.010452955272960097,
                            0.015697952816438227,
                            0.022616985372570498,
                            0.0321816644059754,
                            0.04423303791899698,
                            0.06096950608602681,
                            0.08163607753108654,
                            0.1069855874438356,
                            0.14019446285705825,
                            0.1767215438349438,
                            0.21543255193396638,
                            0.25586031239716955,
                            0.29584767888091074])

xi_dJdt_spin0pt5 = np.asarray([4.963887253352477e-13,
                               0.005100810519721404,
                               0.01050848679362666,
                               0.0166757184827052,
                               0.024045535382879218,
                               0.03334518163020552,
                               0.04541528167553959,
                               0.06122665561800322,
                               0.08199152962185881,
                               0.10932777947133657,
                               0.14433044959985905,
                               0.1850176981735065,
                               0.23241899399459118,
                               0.29031718336895046,
                               0.345749348101726,
                               0.4070831048455746])

xi_dJdt_spin1 = np.asarray([1.4871197939461763e-16,
                            0.007375712752751172,
                            0.01612461091553282,
                            0.027736994548181352,
                            0.04383255580093854,
                            0.06639076949156711,
                            0.09726370081317302,
                            0.1412383020518842,
                            0.20090414817170965,
                            0.28058474271461187,
                            0.38465567525978234,
                            0.5165207324733933,
                            0.681163766666365,
                            0.8869814885281444,
                            1.1205783273694725,
                            1.4009011326594107])

# build splines for each of these functions
xi_dMdt_spin0_spline = InterpolatedUnivariateSpline(astar, xi_dMdt_spin0)
xi_dMdt_spin0pt5_spline = InterpolatedUnivariateSpline(astar, xi_dMdt_spin0pt5)
xi_dMdt_spin1_spline = InterpolatedUnivariateSpline(astar, xi_dMdt_spin1)

xi_dJdt_spin0_spline = InterpolatedUnivariateSpline(astar, xi_dJdt_spin0)
xi_dJdt_spin0pt5_spline = InterpolatedUnivariateSpline(astar, xi_dJdt_spin0pt5)
xi_dJdt_spin1_spline = InterpolatedUnivariateSpline(astar, xi_dJdt_spin1)

BlackMax_greybody_table_5D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi_M': xi_dMdt_spin0_spline, 'xi_J': xi_dJdt_spin0_spline, 'xi-per-dof': True},
  'photon': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'gluon': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'W boson': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'Z boson': {'xi_M': xi_dMdt_spin1_spline, 'xi_J': xi_dJdt_spin1_spline, 'xi-per-dof': True},
  'neutrino': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'electron': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'muon': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'tau': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'up quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'down quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'strange quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'charm quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'bottom quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True},
  'top quark': {'xi_M': xi_dMdt_spin0pt5_spline, 'xi_J': xi_dJdt_spin0pt5_spline, 'xi-per-dof': True}})


# BlackMax doesn't have emission rates from spin 2 particles, so try to estimate using the Kerr ones,
# with their amplitude renormalized so that they match the Friedlander et al. xi emission rate (for mass)
# at zero angular momentum
_xi_dMdt_spin2_rescale = xi0_spin2_5D/xi0_spin2_4D

def _estimated_xi_dMdt_spin2(T_H: float) -> float:
    return _xi_dMdt_spin2_rescale * xi_dMdt_spin2_spline(T_H)

def _estimated_xi_dJdt_spin2(T_H: float) -> float:
    return _xi_dMdt_spin2_rescale * xi_dJdt_spin2_spline(T_H)

BlackMax_graviton_greybody_table_5D = \
 {'5D graviton': {'mass': 0.0, 'dof': 5.0,
                  'xi_M': _estimated_xi_dMdt_spin2, 'xi_J': _estimated_xi_dJdt_spin2, 'xi-per-dof': False}}
