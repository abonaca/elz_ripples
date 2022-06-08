# Third-party packages
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
from scipy.optimize import minimize
import gala.potential as gp
from gala.units import galactic


default_mw = gp.BovyMWPotential2014()
default_disk_bulge = gp.CCompositePotential()
default_disk_bulge['disk'] = default_mw['disk']
default_disk_bulge['bulge'] = default_mw['bulge']


def corot_func(r_cr, Omega, mw_pot):
    vc = mw_pot.circular_velocity([r_cr, 0., 0.])
    return abs(vc - Omega*r_cr * u.kpc).decompose().value[0]


def get_bar_model(Omega, Snlm,
                  alpha=-27*u.deg, m=5e9*u.Msun,
                  mw_pot=None):
    """Get an SCF bar model with the specified pattern speed.

    Parameters
    ----------
    Omega : `~astropy.units.Quantity`
        The bar pattern speed.
    Snlm : array_like
        The expansion coefficients for the bar model.
    alpha : `~astropy.units.Quantity`, optional
        The initial angle of the bar relative to the x-axis.
    m : `~astropy.units.Quantity`, optional
        The bar mass.

    Returns
    -------
    bar : `gala.potential.SCFPotential`
        The bar model with the specified parameters.
    """
    if mw_pot is None:
        mw_pot = default_mw

    res = minimize(corot_func, x0=4., args=(Omega, mw_pot))
    r_cr = res.x[0]
    r_s = r_cr / 3.67 # 3.67 scales this to the value WZ2012 use (60 km/s/kpc)

    return gp.SCFPotential(m=m / 10., r_s=r_s, # 10 is a MAGIC NUMBER: believe
                           Snlm=Snlm,
                           units=galactic,
                           R=rotation_matrix(alpha, 'z'))
