import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.coordinates.matrix_utilities import rotation_matrix

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
from gala.dynamics.nbody import DirectNBody

ham = gp.Hamiltonian(gp.MilkyWayPotential())

import agama
agama.setUnits(mass=1*u.Msun, length=1*u.kpc, velocity=1*u.km/u.s)

import scipy.stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

import pickle
import h5py

import os
from fractions import Fraction
import time

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

#########
# Agama #
#########
def static_disk():
    """"""
    
    print(ham.potential.parameters)
    
    ic_list = initialize_disk()
    ic = np.array([x.value for x in ic_list]).T
    
    pot = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_frozen_nolmc.ini')
    
    # evolve
    timeinit = -0.5  # 3 Gyr back in time - the earliest point in the Sgr simulation
    timecurr =  0.0  # current time is 0, the final point
    times, orbits = agama.orbit(ic=ic, potential=pot, timestart=timeinit, time=timecurr-timeinit, trajsize=100).T
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),100,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    plt.close()
    plt.plot(orbit[:,0,0], orbit[:,0,2], 'k.')
    plt.plot(orbit[:,-1,0], orbit[:,-1,2], 'ro')
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def evolve():
    """"""
    
    ic_list = initialize_disk()
    ic = np.array([x.value for x in ic_list]).T
    
    pot_frozen = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_frozen.ini')
    
    # evolve
    timeinit = -3  # 3 Gyr back in time - the earliest point in the Sgr simulation
    timecurr =  0.0  # current time is 0, the final point
    times, orbits = agama.orbit(ic=ic, potential=pot_frozen, timestart=timecurr, time=timeinit-timecurr, trajsize=100).T
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),100,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    plt.close()
    plt.plot(orbit[:,0,0], orbit[:,0,2], 'k.')
    plt.plot(orbit[:,-1,0], orbit[:,-1,2], 'ro')
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def elz():
    """"""
    
    ic_list = initialize_disk()
    ic = np.array([x.value for x in ic_list]).T
    ic_pos = ic[:,:3]
    ic_vel = ic[:,3:]
    
    pot_frozen = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_frozen.ini')
    
    # evolve
    timeinit = -3  # 3 Gyr back in time - the earliest point in the Sgr simulation
    timecurr =  0.0  # current time is 0, the final point
    times, orbits = agama.orbit(ic=ic, potential=pot_frozen, timestart=timecurr, time=timeinit-timecurr, trajsize=100).T
    
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),100,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    # initial positions
    ic_pos = orbit[:,0,:3]
    ic_vel = orbit[:,0,3:]
    
    epot_init = pot_frozen.potential(ic_pos) * u.km**2*u.s**-2
    ekin_init = (np.linalg.norm(ic_vel, axis=1) * u.km/u.s)**2
    etot_init = (epot_init + ekin_init).to(u.kpc**2*u.Myr**-2)
    l_init = (np.cross(ic_pos, ic_vel) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    
    # final positions
    fc_pos = orbit[:,-1,:3]
    fc_vel = orbit[:,-1,3:]
    
    epot_fin = pot_frozen.potential(fc_pos) * u.km**2*u.s**-2
    ekin_fin = (np.linalg.norm(fc_vel, axis=1) * u.km/u.s)**2
    etot_fin = (epot_fin + ekin_fin).to(u.kpc**2*u.Myr**-2)
    l_fin = (np.cross(fc_pos, fc_vel) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    plt.close()
    plt.plot(l_init[:,2], etot_init, 'r.', mew=0, alpha=0.3)
    plt.plot(l_fin[:,2], etot_fin, 'k.', mew=0, alpha=0.3)
    
    plt.tight_layout()

def elz_live(sgr=True):
    """"""
    
    ic_list = initialize_disk()
    ic = np.array([x.value for x in ic_list]).T
    ic_pos = ic[:,:3]
    ic_vel = ic[:,3:]
    
    if sgr:
        pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_evolving_sgr.ini')
    else:
        pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_evolving.ini')
    
    pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_frozen.ini')
    #pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_frozen_nolmc.ini')
    #pot_evolving = agama.Potential('../data/SCM3.ini')
    
    # evolve
    timeinit = -3  # 3 Gyr back in time - the earliest point in the Sgr simulation
    timecurr =  0.0  # current time is 0, the final point
    Ntraj = 100
    times, orbits = agama.orbit(ic=ic, potential=pot_evolving, timestart=timeinit, time=timecurr-timeinit, trajsize=Ntraj).T
    
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),Ntraj,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    #np.savez('../data/model_sgr.{:d}.npz'.format(sgr), orbit=orbit, t=times)
    
    # initial positions
    ic_pos = orbit[:,0,:3]
    ic_vel = orbit[:,0,3:]
    
    epot_init = pot_evolving.potential(ic_pos, t=timeinit) * u.km**2*u.s**-2
    ekin_init = (np.linalg.norm(ic_vel, axis=1) * u.km/u.s)**2
    etot_init = (epot_init + ekin_init).to(u.kpc**2*u.Myr**-2)
    l_init = (np.cross(ic_pos, ic_vel) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    
    # final positions
    fc_pos = orbit[:,-1,:3]
    fc_vel = orbit[:,-1,3:]
    #fc_pos = orbit[:,0,:3]
    #fc_vel = orbit[:,0,3:]
    
    epot_fin = pot_evolving.potential(fc_pos, t=timecurr) * u.km**2*u.s**-2
    ekin_fin = (np.linalg.norm(fc_vel, axis=1) * u.km/u.s)**2
    etot_fin = (epot_fin + ekin_fin).to(u.kpc**2*u.Myr**-2)
    l_fin = (np.cross(fc_pos, fc_vel) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    print(1-etot_init/etot_fin)
    
    plt.close()
    plt.plot(l_init[:,2], etot_init, 'r.', mew=0, alpha=0.3)
    plt.plot(l_fin[:,2], etot_fin, 'k.', mew=0, alpha=0.3)
    
    #plt.xlim(-5,0)
    #plt.ylim(-0.14, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    #plt.savefig('../plots/model_elz_sgr.{:d}.png'.format(sgr))
    #plt.savefig('../plots/model_elz_static.png')

def observe_model(sgr=False):
    """"""
    m = np.load('../data/model_sgr.{:d}.npz'.format(sgr))
    orbit = m['orbit']
    
    if sgr:
        pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_evolving_sgr.ini')
    else:
        pot_evolving = agama.Potential('../data/Sgr_stream_simulations/potentials_triax/potential_evolving.ini')
    
    timecurr =  0.0
    
    # final positions
    fc_pos = orbit[:,-1,:3]
    fc_vel = orbit[:,-1,3:]
    
    epot_fin = pot_evolving.potential(fc_pos, t=timecurr) * u.km**2*u.s**-2
    ekin_fin = (np.linalg.norm(fc_vel, axis=1) * u.km/u.s)**2
    etot_fin = (epot_fin + ekin_fin).to(u.kpc**2*u.Myr**-2)
    l_fin = (np.cross(fc_pos, fc_vel) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    #
    ctrue = coord.SkyCoord(x=fc_pos[:,0]*u.kpc, y=fc_pos[:,1]*u.kpc, z=fc_pos[:,2]*u.kpc, v_x=fc_vel[:,0]*u.km/u.s, v_y=fc_vel[:,1]*u.km/u.s, v_z=fc_vel[:,2]*u.km/u.s, frame=gc_frame)
    ctrue_eq = ctrue.transform_to('icrs')
    
    cobs_eq = coord.SkyCoord(ra=ctrue_eq.ra, dec=ctrue_eq.dec, distance=ctrue_eq.distance*(1 + np.random.randn(len(ctrue_eq))*0.1), pm_ra_cosdec=ctrue_eq.pm_ra_cosdec, pm_dec=ctrue_eq.pm_dec, radial_velocity=ctrue_eq.radial_velocity)
    cobs = cobs_eq.transform_to(gc_frame)
    
    fc_pos_obs = np.array([cobs.x.to(u.kpc).value, cobs.y.to(u.kpc).value, cobs.z.to(u.kpc).value]).T
    fc_vel_obs = np.array([cobs.v_x.to(u.km/u.s).value, cobs.v_y.to(u.km/u.s).value, cobs.v_z.to(u.km/u.s).value]).T
    
    epot_fin_obs = pot_evolving.potential(fc_pos_obs, t=timecurr) * u.km**2*u.s**-2
    ekin_fin_obs = (np.linalg.norm(fc_vel_obs, axis=1) * u.km/u.s)**2
    etot_fin_obs = (epot_fin_obs + ekin_fin_obs).to(u.kpc**2*u.Myr**-2)
    l_fin_obs = (np.cross(fc_pos_obs, fc_vel_obs) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    
    
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    
    plt.sca(ax[0])
    #plt.plot(l_fin[:,2], etot_fin, 'k.', ms=2, mew=0, alpha=0.3)
    plt.plot(l_fin_obs[:,2], etot_fin_obs, 'k.', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(-5,5)
    plt.ylim(-0.14, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    #plt.plot(l_fin_obs[:,2], etot_fin_obs, 'k.', ms=2, mew=0, alpha=0.3)
    plt.hist(etot_fin_obs.value, bins=np.linspace(-0.12,-0.04,50))
    
    #plt.xlim(-6,6)
    #plt.ylim(-0.14, -0.02)
    #plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    #plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()


def initialize_ndisk(Nrand=50000, seed=2572, ret=True, graph=False):
    """Draw particles from N21 disk"""
    
    td = pickle.load(open('../data/thick_disk.pkl', 'rb'))
    eta = 1.4
    eta_x = 2.5
    eta_v = 1.6
    
    #call = coord.Galactocentric(x=-1*td['x'][:,0]*u.kpc, y=td['x'][:,1]*u.kpc, z=td['x'][:,2]*u.kpc, v_x=-1*eta*td['v'][:,0]*u.km/u.s, v_y=eta*td['v'][:,1]*u.km/u.s, v_z=eta*td['v'][:,2]*u.km/u.s)
    call = coord.Galactocentric(x=-1*eta_x*td['x'][:,0]*u.kpc, y=eta_x*td['x'][:,1]*u.kpc, z=eta_x*td['x'][:,2]*u.kpc, v_x=-1*eta_v*td['v'][:,0]*u.km/u.s, v_y=eta_v*td['v'][:,1]*u.km/u.s, v_z=eta_v*td['v'][:,2]*u.km/u.s)
    
    # select stars with high energy
    w0 = gd.PhaseSpacePosition(call.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0]
    ind_highe = etot>-0.18*u.kpc**2*u.Myr**-2
    call = call[ind_highe]
    
    ## H3 selection function
    #cgal = call.transform_to(coord.Galactic())
    #ind_h3 = (np.abs(cgal.b)>20*u.deg) & (cgal.distance>2*u.kpc)
    ##ind_h3 = (cgal.distance>2*u.kpc)
    #call = call[ind_h3]
    
    # pick randomly Nrand stars
    np.random.seed(seed)
    Ntot = np.size(call.x)
    ind = np.arange(0, Ntot, 1, dtype=int)
    if Nrand<Ntot:
        ind_rand = np.random.choice(ind, size=Nrand, replace=False)
    else:
        ind_rand = ind
    c = call[ind_rand]
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0].value
    lz = orbit.angular_momentum()[2][0]
    
    if graph:
        print(np.sum((etot>-0.16) & (etot<-0.06)))
        print(np.sum(etot<0))
        
        t = Table.read('../data/rcat_giants.fits')
        ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
        ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) # & (t['FeH']<-1)
        #t = t[ind_disk]
        
        ebins = np.linspace(-0.30,-0.02,200)
        #ebins = np.linspace(-0.16,-0.06,50)
        
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(lz, etot, 'k.', ms=3, alpha=0.1, mew=0, zorder=0)
        plt.plot(t['Lz'], t['E_tot_pot1'], 'o', color='tab:orange', zorder=0, ms=1, alpha=0.1)
        
        plt.xlim(-6,6)
        plt.ylim(-0.3, -0.02)
        plt.ylim(-0.18, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
        plt.sca(ax[1])
        #plt.hist(lz.value, bins=100, density=True, label='Model')
        plt.hist(etot, bins=ebins, density=True, label='Model')
        plt.hist(t['E_tot_pot1'][ind_disk], bins=ebins, density=True, histtype='step', label='H3', lw=2)
        
        plt.legend(fontsize='small')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
        plt.tight_layout()
        plt.savefig('../plots/initialize_ndisk_phase.png')
    
    if ret:
        ind = etot<0
        return c[ind]

def initialize_nhalo(Nrand=50000, seed=2572, ret=True, graph=False):
    """"""
    
    th = Table.read('/home/ana/data/gse/gse_stars.fits')
    eta = 1.
    eta_x = 1.
    eta_v = 1.
    
    call = coord.Galactocentric(x=eta_x*th['X']*u.kpc, y=eta_x*th['Y']*u.kpc, z=eta_x*th['Z']*u.kpc, v_x=eta_v*th['Vx']*u.km/u.s, v_y=eta_v*th['Vy']*u.km/u.s, v_z=eta_v*th['Vz']*u.km/u.s)
    
    # select stars with high energy
    w0 = gd.PhaseSpacePosition(call.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0]
    ind_highe = etot>-0.18*u.kpc**2*u.Myr**-2
    call = call[ind_highe]
    
    ## H3 selection function
    #cgal = call.transform_to(coord.Galactic())
    #ind_h3 = (np.abs(cgal.b)>20*u.deg) & (cgal.distance>2*u.kpc)
    ##ind_h3 = (cgal.distance>2*u.kpc)
    #call = call[ind_h3]
    
    # pick randomly Nrand stars
    np.random.seed(seed)
    Ntot = np.size(call.x)
    ind = np.arange(0, Ntot, 1, dtype=int)
    if Nrand<Ntot:
        ind_rand = np.random.choice(ind, size=Nrand, replace=False)
    else:
        ind_rand = ind
    c = call[ind_rand]
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0].value
    lz = orbit.angular_momentum()[2][0]
    
    if graph:
        print(np.sum((etot>-0.16) & (etot<-0.06)))
        print(np.sum(etot<0))
        
        t = Table.read('../data/rcat_giants.fits')
        ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
        ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) #& (t['SNR']>5) # & (t['FeH']<-1)
        #t = t[ind_gse]
        
        #ebins = np.linspace(-0.30,-0.02,200)
        ebins = np.linspace(-0.16,-0.06,100)
        
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(lz, etot, 'k.', ms=3, alpha=0.1, mew=0, zorder=0)
        plt.plot(t['Lz'], t['E_tot_pot1'], 'o', color='tab:orange', zorder=0, ms=1, alpha=0.1)
        
        plt.xlim(-6,6)
        plt.ylim(-0.3, -0.02)
        plt.ylim(-0.18, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
        plt.sca(ax[1])
        #plt.hist(lz.value, bins=100, density=True, label='Model')
        plt.hist(etot, bins=ebins, density=True, label='Model')
        plt.hist(t['E_tot_pot1'][ind_gse], bins=ebins, density=True, histtype='step', label='H3', lw=1)
        
        plt.legend(fontsize='small')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
        plt.tight_layout()
        plt.savefig('../plots/initialize_nhalo_phase.png')
    
    if ret:
        ind = etot<0
        return c[ind]

def mwpotential():
    pot = agama.GalaPotential(
        dict(type='miyamotonagai', mass=6.8e10, scaleradius=3.0, scaleheight=0.28),  # disk
        dict(type='dehnen', mass=5.00e9, scaleradius=1.0),   # bulge
        dict(type='dehnen', mass=1.71e9, scaleradius=0.07),  # nucleus
        dict(type='nfw',    mass=5.4e11, scaleradius=15.62)) # halo
    
    c = initialize_ndisk(Nrand=50000)
    ic = np.array([c.x.value, c.y.value, c.z.value, c.v_x.value, c.v_y.value, c.v_z.value]).T
    
    timeinit = -5  # 3 Gyr back in time - the earliest point in the Sgr simulation
    timecurr =  0.0  # current time is 0, the final point
    times, orbits = agama.orbit(ic=ic, potential=pot, timestart=timecurr, time=timeinit-timecurr, trajsize=100).T
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),100,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    plt.close()
    plt.plot(orbit[:,0,0], orbit[:,0,2], 'k.')
    plt.plot(orbit[:,-1,0], orbit[:,-1,2], 'ro')
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()



