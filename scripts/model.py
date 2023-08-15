import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.constants import G

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
from gala.dynamics.nbody import DirectNBody
import gala.integrate as gi

import interact3 as interact

#ham = gp.Hamiltonian(gp.MilkyWayPotential())
mwpot = gp.load('/home/ana/data/MilkyWayPotential2022.yml')
ham = gp.Hamiltonian(mwpot)


import scipy.stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy import ndimage
from scipy.signal import find_peaks

from functools import partial

import pickle
import h5py

import os
from fractions import Fraction
import time

import readsnap
import agama
from configparser import RawConfigParser

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()
solarRadius = gc_frame.galcen_distance.value


def initialize_idisk(ret=True, graph=False, Ntot=1000, Nr=20, seed=193, sigma_z=0*u.kpc, sigma_vz=0*u.km/u.s, verbose=False):
    """Initial positions and velocities of stars in an idealized disk: uniform, circular, and thin"""
    
    # ensure sufficient number of stars per radius
    if Ntot<Nr*20:
        Ntot = Nr * 20
    
    # place discrete radii
    radii = np.logspace(np.log10(2),np.log10(50),Nr) * u.kpc
    
    # number of particles per radial bin
    Ntmin = 15
    p = scipy.stats.norm.pdf(radii.value, loc=12, scale=8)
    #p = scipy.stats.norm.logpdf(np.log10(radii.value), loc=np.log10(11), scale=6)
    Nt = np.int64(Ntot * p / np.sum(p))
    ind = Nt<Ntmin
    Nt[ind] = Ntmin
    
    # size of an arc
    circumference = 2*np.pi*radii
    arc = circumference / Nt
    
    N = np.sum(Nt)
    np.random.seed(seed)
    if verbose:
        print(N, Ntot)
    
    x = np.zeros(N) * u.kpc
    y = np.zeros(N) * u.kpc
    z = np.zeros(N) * u.kpc
    z = np.random.randn(N) * sigma_z
    
    vx = np.zeros(N) * u.km/u.s
    vy = np.zeros(N) * u.km/u.s
    vz = np.zeros(N) * u.km/u.s
    vz = np.random.randn(N) * sigma_vz
    
    q = np.array([radii.value, np.zeros(Nr), np.zeros(Nr)]) * u.kpc
    vnorm = -1*ham.potential.circular_velocity(q)

    k = 0
    for i in range(Nr):
        for j in range(Nt[i]):
            theta = j * (arc[i]/radii[i]).decompose().value * u.radian
            
            x[k] = radii[i] * np.cos(theta)
            y[k] = radii[i] * np.sin(theta)
            
            vx[k] = vnorm[i] * (-np.sin(theta))
            vy[k] = vnorm[i] * np.cos(theta)
            
            k += 1
    
    c = coord.Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
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
        t = t[ind_disk]
        
        ebins = np.linspace(-0.20,-0.02,200)
        
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(lz, etot, 'k.', ms=1, alpha=0.1)
        
        plt.xlim(-8,2)
        plt.ylim(-0.2, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
        plt.sca(ax[1])
        plt.hist(etot, bins=ebins, density=True, label='Model')
        plt.hist(t['E_tot_pot1'], bins=ebins, density=True, histtype='step', label='H3', lw=2)
        
        plt.legend(fontsize='small')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
        plt.tight_layout()
        plt.savefig('../plots/initialize_idisk_phase.png')
        
    if ret:
        ind = etot<0
        return c[ind]

def initialize_disk(graph=False, gala=False, logspace=False, Nr=20, const_arc=True, sigma_z=0*u.kpc, sigma_vz=0*u.km/u.s, verbose=True):
    """Initial positions and velocities of stars in a uniform, circular, thin disk"""
    
    if logspace:
        radii = np.logspace(0.7,1.5,Nr) * u.kpc
        radii = np.logspace(0.7,1.6,Nr) * u.kpc
        radii = np.logspace(0.6,1.6,Nr) * u.kpc
        radii = np.logspace(np.log10(2),np.log10(50),Nr) * u.kpc
    else:
        radii = np.linspace(5,30,Nr) * u.kpc
        radii = np.linspace(4,40,Nr) * u.kpc
        radii = np.linspace(2,50,Nr) * u.kpc
    
    circumference = 2*np.pi*radii
    Nr = np.size(radii)
    Ntot = Nr * 20

    if const_arc:
        arc = np.ones(Nr) * 1*u.kpc
        
        #Nt = np.int64(np.ceil(circumference/arc).decompose().value)
        Nt = np.int64((circumference/arc).decompose().value)
    else:
        #Nt = np.ones(Nr, dtype=int) * 30
        Ntmin = 15
        #p = scipy.stats.norm.pdf(radii.value, loc=10, scale=3)
        p = scipy.stats.norm.pdf(radii.value, loc=11, scale=6)
        Nt = np.int64(Ntot * p / np.sum(p))
        ind = Nt<Ntmin
        Nt[ind] = Ntmin
        
        arc = circumference / Nt
    
    N = np.sum(Nt)
    np.random.seed(421)
    if verbose:
        print(N, Ntot)
    
    x0 = np.zeros(N) * u.kpc
    y0 = np.zeros(N) * u.kpc
    z0 = np.zeros(N) * u.kpc
    z0 = np.random.randn(N) * sigma_z
    
    vx0 = np.zeros(N) * u.km/u.s
    vy0 = np.zeros(N) * u.km/u.s
    vz0 = np.zeros(N) * u.km/u.s
    vz0 = np.random.randn(N) * sigma_vz
    
    if gala:
        q = np.array([radii.value, np.zeros(Nr), np.zeros(Nr)]) * u.kpc
        vnorm = -1*ham.potential.circular_velocity(q)
    else:
        vnorm = np.ones(Nr) * (-200*u.km/u.s)
    
    k = 0
    for i in range(Nr):
        for j in range(Nt[i]):
            theta = j * (arc[i]/radii[i]).decompose().value * u.radian
            
            x0[k] = radii[i] * np.cos(theta)
            y0[k] = radii[i] * np.sin(theta)
            
            vx0[k] = vnorm[i] * (-np.sin(theta))
            vy0[k] = vnorm[i] * np.cos(theta)
            
            k += 1
    
    ic = [x0, y0, z0, vx0, vy0, vz0]
    
    if graph:
        plt.close()
        plt.plot(x0, y0, 'ko')
        
        plt.gca().set_aspect('equal')
        plt.tight_layout()
    
    return ic



def format_gse_dm():
    """"""
    
    t = Table.read('../data/GSE_DM_0010.fits.gz')
    colnames = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
    
    for k in colnames:
        t.rename_column('{:s}_gal'.format(k), k)
    
    t.write('/home/ana/data/gse_dm.fits', overwrite=True)

def analyze_gse():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) & (t['Rapo_pot1']<50)
    t = t[ind_gse]
    
    q = np.array([t['X_gal'], t['Y_gal'], t['Z_gal']]) * u.kpc
    vcirc = ham.potential.circular_velocity(q)
    
    vtot = np.sqrt(t['Vr_gal']**2 + t['Vtheta_gal']**2 + t['Vphi_gal']**2)
    vr = t['Vr_gal'] / vtot
    vtheta = t['Vtheta_gal'] / vtot
    vphi = t['Vphi_gal'] / vtot
    
    #print(t.colnames)
    
    Nrand = 50000
    logr = np.random.randn(Nrand)*0.23 + 1.3
    logr = np.random.randn(Nrand)*0.18 + 1.13
    logr = np.random.randn(Nrand)*0.2 + 1.2
    r = 10**logr
    bins = np.linspace(5,50,30)
    
    logv = np.random.randn(Nrand)*0.34 + 2.14
    v = 10**logv
    vbins = np.linspace(0,400,30)
    
    vth = np.abs(np.random.randn(Nrand)*0.32)
    vphi = np.abs(np.random.randn(Nrand)*0.32)
    vrad = np.sqrt(np.abs(1 - vth**2 - vphi**2))
    thbins = np.linspace(0,1,30)
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    plt.sca(ax[0])
    #plt.hist(t['Rapo_pot1'], bins=bins, density=True)
    plt.hist(t['R_gal'], bins=bins, density=True, label='H3')
    plt.hist(r, bins=bins, density=True, histtype='step', color='k', label='Model')
    
    plt.legend(fontsize='small')
    plt.xlabel('$R_{gal}$ [kpc]')
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    plt.hist(vtot, bins=vbins, density=True)
    plt.hist(v, bins=vbins, density=True, histtype='step', color='k')

    plt.xlabel('$V_{tot}$ [km s$^{-1}$]')
    plt.ylabel('Density')
    
    plt.sca(ax[2])
    plt.hist(np.abs(vr), bins=thbins, density=True, label='$V_r$')
    plt.hist(np.abs(vtheta), bins=thbins, alpha=0.3, density=True, label='$V_\\theta$')
    plt.hist(np.abs(vphi), bins=thbins, alpha=0.3, density=True, label='$V_\phi$')
    plt.hist(vth, bins=thbins, density=True, histtype='step', color='k', label='')
    plt.hist(vphi, bins=thbins, density=True, histtype='step', color='k', label='')
    plt.hist(vrad, bins=thbins, density=True, histtype='step', color='k', label='')
    
    plt.legend(fontsize='small')
    plt.xlabel('|$V_x$ / $V_{tot}$|')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('../plots/initialize_halo.png')

def initialize_halo(Nrand=50000, seed=2924, ret=True, graph=False):
    """Draw halo to match present-day positions and velocities of GSE stars in H3"""
    
    #Nrand = 50000
    np.random.seed(seed)
    
    # positions
    phi = np.random.rand(Nrand)*2*np.pi
    costheta = np.random.rand(Nrand)*2 - 1
    theta = np.arccos(costheta)
    logr = np.random.randn(Nrand)*0.18 + 1.13
    logr = np.random.randn(Nrand)*0.2 + 1.2
    r = 10**logr * u.kpc
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # velocities
    logv = np.random.randn(Nrand)*0.34 + 2.14
    v = 10**logv * u.km/u.s
    vtheta = np.random.randn(Nrand)*0.32
    vphi = np.random.randn(Nrand)*0.32
    vrad = np.sqrt(np.abs(1 - vtheta**2 - vphi**2))
    
    vx = v * (vrad*np.sin(theta)*np.cos(phi) + vtheta*np.cos(theta)*np.cos(phi) - vphi*np.sin(phi))
    vy = v * (vrad*np.sin(theta)*np.sin(phi) + vtheta*np.cos(theta)*np.sin(phi) + vphi*np.cos(phi))
    vz = v * (vrad*np.cos(theta) - vtheta*np.sin(theta))
    
    c = coord.Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
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
        
        ebins = np.linspace(-0.20,-0.02,50)
        #ebins = np.linspace(-0.16,-0.06,50)
        
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(lz, etot, 'k.', ms=2, alpha=0.3)
        
        plt.xlim(-6,6)
        plt.ylim(-0.2, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        
        plt.sca(ax[1])
        plt.hist(etot, bins=ebins, density=True, label='Model')
        plt.hist(t['E_tot_pot1'], bins=ebins, density=True, histtype='step', lw=2, label='H3')
        
        plt.legend(fontsize='small')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
        
        plt.tight_layout()
        plt.savefig('../plots/initialize_halo_phase.png')
    
    if ret:
        ind = etot<0
        return c[ind]


def analyze_td():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    #ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) & (t['Rapo_pot1']<50)
    t = t[ind_disk]
    
    q = np.array([t['X_gal'], t['Y_gal'], t['Z_gal']]) * u.kpc
    vcirc = ham.potential.circular_velocity(q)
    
    vtot = np.sqrt(t['Vr_gal']**2 + t['Vtheta_gal']**2 + t['Vphi_gal']**2)
    vr = t['Vr_gal'] / vtot
    vtheta = t['Vtheta_gal'] / vtot
    vphi = t['Vphi_gal'] / vtot
    
    rho = np.sqrt(t['X_gal']**2 + t['Y_gal']**2)
    #print(t.colnames)
    
    Nrand = 50000
    logr = np.random.randn(Nrand)*0.18 + 1.
    #logr = np.random.randn(Nrand)*0.22 + 1.13
    r = 10**logr
    bins = np.linspace(2,40,30)
    
    logz = np.random.randn(Nrand)*0.17 + 0.54
    z = 10**logz
    zbins = np.linspace(0,15,30)
    
    v = np.random.randn(Nrand)*50 + 190
    vbins = np.linspace(0,400,30)
    
    vth_ = np.abs(np.random.randn(Nrand)*0.22)
    vrad_ = np.abs(np.random.randn(Nrand)*0.37)
    vphi_ = np.sqrt(np.abs(1 - vth_**2 - vrad_**2))
    thbins = np.linspace(0,1,30)
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    
    plt.sca(ax[0])
    plt.hist(rho, bins=bins, density=True, label='H3')
    plt.hist(r, bins=bins, density=True, histtype='step', color='k', label='Model')
    
    plt.legend(fontsize='small')
    plt.xlabel('$\\rho$ [kpc]')
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    plt.hist(np.abs(t['Z_gal']), bins=zbins, density=True)
    plt.hist(z, bins=zbins, density=True, histtype='step', color='k')
    
    plt.xlabel('Z [kpc]')
    plt.ylabel('Density')
    
    plt.sca(ax[2])
    plt.hist(vtot, bins=vbins, density=True)
    plt.hist(v, bins=vbins, density=True, histtype='step', color='k')

    plt.xlabel('$V_{tot}$ [km s$^{-1}$]')
    plt.ylabel('Density')
    
    plt.sca(ax[3])
    plt.hist(np.abs(vphi), bins=thbins, density=True, label='$V_\phi$')
    plt.hist(np.abs(vtheta), bins=thbins, alpha=0.3, density=True, label='$V_\\theta$')
    plt.hist(np.abs(vr), bins=thbins, alpha=0.3, density=True, label='$V_r$')
    plt.hist(vth_, bins=thbins, density=True, histtype='step', color='0.5', label='')
    plt.hist(vphi_, bins=thbins, density=True, histtype='step', color='k', label='')
    plt.hist(vrad_, bins=thbins, density=True, histtype='step', color='k', label='')
    
    plt.legend(fontsize='small')
    plt.xlabel('|$V_x$ / $V_{tot}$|')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('../plots/initialize_td.png')

def initialize_td(Nrand=50000, seed=2924, ret=True, graph=False):
    """Draw disk to match present-day positions and velocities of disk stars in H3"""
    
    np.random.seed(seed)
    
    # positions
    phi = np.random.rand(Nrand)*2*np.pi
    logr = np.random.randn(Nrand)*0.18 + 1.
    #logr = np.random.randn(Nrand)*0.25 + 1.11
    rho = 10**logr * u.kpc
    logz = np.random.randn(Nrand)*0.17 + 0.54
    #logz = np.random.randn(Nrand)*0.2 + 0.55
    signz = 2*np.random.randint(0, 2, size=Nrand) - 1
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = signz * 10**logz * u.kpc
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    
    # velocities
    v = (np.random.randn(Nrand)*50 + 190) * u.km/u.s
    vtheta = np.abs(np.random.randn(Nrand)*0.22)
    vrad = np.abs(np.random.randn(Nrand)*0.37)
    vphi = -1*np.sqrt(np.abs(1 - vtheta**2 - vrad**2))
    
    vx = v * (vrad*np.sin(theta)*np.cos(phi) + vtheta*np.cos(theta)*np.cos(phi) - vphi*np.sin(phi))
    vy = v * (vrad*np.sin(theta)*np.sin(phi) + vtheta*np.cos(theta)*np.sin(phi) + vphi*np.cos(phi))
    vz = v * (vrad*np.cos(theta) - vtheta*np.sin(theta))
    
    c = coord.Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
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
        t = t[ind_disk]
        
        ebins = np.linspace(-0.20,-0.02,200)
        #ebins = np.linspace(-0.16,-0.06,50)
        
        plt.close()
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        
        plt.sca(ax[0])
        plt.plot(lz, etot, 'k.', ms=1, alpha=0.1)
        
        plt.xlim(-8,2)
        plt.ylim(-0.2, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
        plt.sca(ax[1])
        plt.hist(etot, bins=ebins, density=True, label='Model')
        #plt.hist(t['E_tot_pot1'], bins=ebins, density=True, histtype='step', label='H3', lw=2)
        
        plt.legend(fontsize='small')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
        plt.tight_layout()
        plt.savefig('../plots/initialize_td_phase.png')
    
    if ret:
        ind = etot<0
        return c[ind]


###########
# Nbody GSE

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
        #plt.plot(t['Lz'], t['E_tot_pot1'], 'o', color='tab:orange', zorder=0, ms=1, alpha=0.1)
        
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


##################
# Equilibrium disk

def nbody_disk():
    """"""
    filename = '../data/snap_010'
    
    posbulge = readsnap.read_block(filename, 'POS ', parttype=3) # load positions of halo particles
    velbulge = readsnap.read_block(filename, 'VEL ', parttype=3) # load velocities of halo particles
    idbulge = readsnap.read_block(filename, 'ID  ', parttype=3) # load IDs of halo particles
    
    posdisk = readsnap.read_block(filename, 'POS ', parttype=2) # load positions of disk particles
    veldisk = readsnap.read_block(filename, 'VEL ', parttype=2) # load velocities of disk particles
    iddisk = readsnap.read_block(filename, 'ID  ', parttype=2) # load IDs of disk particles
    
    posdisk = posdisk - np.mean(posbulge, axis=0)
    veldisk = veldisk - np.mean(velbulge, axis=0)
    
    print(posdisk)
    print(veldisk)
    
    outdict = dict(x=posdisk, v=veldisk)
    pickle.dump(outdict, open('../data/thick_disk.pkl', 'wb'))
    
    plt.close()
    plt.figure()

def plot_td():
    """"""
    td = pickle.load(open('../data/thick_disk.pkl', 'rb'))
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.scatter(td['x'][:,0], td['x'][:,1], c=td['v'][:,1])
    plt.xlabel('X [kpc]')
    plt.ylabel('Y [kpc]')
    plt.gca().set_aspect('equal')
    
    plt.sca(ax[1])
    plt.plot(td['x'][:,0], td['x'][:,2], 'k.', mew=0, alpha=0.1, ms=2)
    plt.xlabel('X [kpc]')
    plt.ylabel('Z [kpc]')
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    plt.savefig('../plots/nbody_disk_init.png')

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

# from agama
# display some information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    #densBulge= model.components[1].getDensity()
    #densHalo = model.components[2].getDensity()
    pt0 = (solarRadius, 0, 0)
    pt1 = (solarRadius, 0, 1)
    print("Disk total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densDisk.totalMass(), densDisk.density(pt0)*1e-9, densDisk.density(pt1)*1e-9))  # per pc^3, not kpc^3
    #print("Halo total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        #(densHalo.totalMass(), densHalo.density(pt0)*1e-9, densHalo.density(pt1)*1e-9))
    print("Potential at origin=-(%g km/s)^2, total mass=%g Msun" % \
        ((-model.potential.potential(0,0,0))**0.5, model.potential.totalMass()))
    densDisk.export ("dens_disk_" +iteration);
    #densBulge.export("dens_bulge_"+iteration);
    #densHalo.export ("dens_halo_" +iteration);
    model.potential.export("potential_"+iteration);
    #writeRotationCurve("rotcurve_"+iteration, (model.potential[1],  # disk potential (CylSpline)
        #agama.Potential(type='Multipole', lmax=6, density=densBulge),        # -"- bulge
        #agama.Potential(type='Multipole', lmax=6, density=densHalo) ) )      # -"- halo

def agama_disk():
    """Set up a self-consistent disk with agama"""
    
    iniPotenThickDisk = dict(
        type = 'Disk',
        surfaceDensity = 3.0e+8,
        scaleRadius = 2.1,
        scaleHeight = -0.7
        )
    
    iniDFThickDisk = dict(
        type = 'Exponential',
        mass = .95e10,
        Jr0 = 63,
        Jz0 = 30,
        Jphi0 = 400,
        pr = 0.13,
        pz = 0.05,
        addJden = 20,
        addJvel = 40)
    
    ### parameters of the bulge+disk+stellar halo component of self-consistent model
    # definition of grid in cylindrical radius - radii in kpc
    iniSCMDisk = dict(
        RminCyl = 0.1,
        RmaxCyl = 30,
        sizeRadialCyl = 20,
        # definition of grid in vertical direction
        zminCyl = 0.05,
        zmaxCyl = 10,
        sizeVerticalCyl= 16,
        )
    

    ### parameters for the potential solvers for the entire self-consistent model
    # definition of spherical grid for constructing multipole potential expansion;
    # this grid should encompass that of the halo component, but also should resolve
    # the bulge density profile at small radii; meaning of parameters is the same
    iniSCM = dict(
        rminSph = 0.01,
        rmaxSph = 1000.,
        sizeRadialSph = 50,
        lmaxAngularSph = 4,
        RminCyl = 0.1,
        RmaxCyl = 50,
        sizeRadialCyl = 30,
        zminCyl = 0.04,
        zmaxCyl = 20,
        sizeVerticalCyl = 30,
        useActionInterpolation = False
        )
    
    # define external unit system describing the data (including the ini parameters)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun
    
    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    #densityBulge       = agama.Density(**iniPotenBulge)
    #densityDarkHalo    = agama.Density(**iniPotenDarkHalo)
    #densityThinDisk    = agama.Density(**iniPotenThinDisk)
    densityThickDisk   = agama.Density(**iniPotenThickDisk)
    #densityGasDisk     = agama.Density(**iniPotenGasDisk)
    #densityStellarDisk = agama.Density(densityThinDisk, densityThickDisk)  # composite
    #densityStellarDisk = agama.Density(densityThickDisk)  # composite
    densityStellarDisk = densityThickDisk

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityStellarDisk, disklike=True))
    #model.components.append(agama.Component(density=densityBulge,       disklike=False))
    #model.components.append(agama.Component(density=densityDarkHalo,    disklike=False))
    #model.components.append(agama.Component(density=densityGasDisk,     disklike=True))

    # compute the initial potential
    model.iterate()
    printoutInfo(model, 'init')
    
    Gpot=agama.GalaPotential(ham.potential)
    
    dfThickDisk = agama.DistributionFunction(potential=Gpot, **iniDFThickDisk)
    #dfStellar = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo)
    dfStellar = dfThickDisk
    model.components[0] = agama.Component(df=dfStellar, disklike=True, **iniSCMDisk)
    
    xv = agama.GalaxyModel(potential=model.potential, df=dfThickDisk, af=model.af).sample(20000)[0]
    
    cgal = coord.Galactocentric(x=xv[:,0]*u.kpc, y=xv[:,1]*u.kpc, z=xv[:,2]*u.kpc, v_x=xv[:,3]*u.km/u.s, v_y=xv[:,4]*u.km/u.s, v_z=xv[:,5]*u.km/u.s)
    cg = cgal.transform_to(coord.Galactic())
    ind_h3 = (np.abs(cg.b)>30*u.deg) & (cg.distance>2*u.kpc)
    print(np.sum(ind_h3))
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    #plt.plot(xv[:,0], xv[:,1], 'k.')
    #plt.gca().set_aspect('equal', adjustable='datalim')
    plt.hist(cgal.cylindrical.rho.value, bins=np.linspace(0,50,30), log=True)
    
    plt.sca(ax[1])
    #plt.plot(xv[:,0], xv[:,2], 'k.')
    #plt.gca().set_aspect('equal', adjustable='datalim')
    plt.hist(cgal.z.value, bins=np.linspace(0,10,30), log=True)
    
    plt.tight_layout()

def agama_model():
    """Agama model for the entire Milky Way"""
    
    iniFileName = "../data/SCM_MW.ini"
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenThinDisk = dict(ini.items("Potential thin disk"))
    iniPotenThickDisk= dict(ini.items("Potential thick disk"))
    iniPotenGasDisk  = dict(ini.items("Potential gas disk"))
    iniPotenBulge    = dict(ini.items("Potential bulge"))
    iniPotenDarkHalo = dict(ini.items("Potential dark halo"))
    iniDFThinDisk    = dict(ini.items("DF thin disk"))
    iniDFThickDisk   = dict(ini.items("DF thick disk"))
    iniDFStellarHalo = dict(ini.items("DF stellar halo"))
    iniDFDarkHalo    = dict(ini.items("DF dark halo"))
    iniDFBulge       = dict(ini.items("DF bulge"))
    iniSCMHalo       = dict(ini.items("SelfConsistentModel halo"))
    iniSCMBulge      = dict(ini.items("SelfConsistentModel bulge"))
    iniSCMDisk       = dict(ini.items("SelfConsistentModel disk"))
    iniSCM           = dict(ini.items("SelfConsistentModel"))
    solarRadius      = ini.getfloat("Data", "SolarRadius")

    # define external unit system describing the data (including the parameters in INI file)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityBulge       = agama.Density(**iniPotenBulge)
    densityDarkHalo    = agama.Density(**iniPotenDarkHalo)
    densityThinDisk    = agama.Density(**iniPotenThinDisk)
    densityThickDisk   = agama.Density(**iniPotenThickDisk)
    densityGasDisk     = agama.Density(**iniPotenGasDisk)
    densityStellarDisk = agama.Density(densityThinDisk, densityThickDisk)  # composite

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityStellarDisk, disklike=True))
    model.components.append(agama.Component(density=densityBulge,       disklike=False))
    model.components.append(agama.Component(density=densityDarkHalo,    disklike=False))
    model.components.append(agama.Component(density=densityGasDisk,     disklike=True))

    # compute the initial potential
    model.iterate()
    printoutInfo(model, "init")

    print("\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: " \
        "Mdisk=%g Msun, Mbulge=%g Msun, Mhalo=%g Msun, Mgas=%g Msun" % \
        (densityStellarDisk.totalMass(), densityBulge.totalMass(), \
        densityDarkHalo.totalMass(), densityGasDisk.totalMass()))

    # create the dark halo DF
    dfHalo  = agama.DistributionFunction(potential=model.potential, **iniDFDarkHalo)
    # same for the bulge
    dfBulge = agama.DistributionFunction(potential=model.potential, **iniDFBulge)
    # same for the stellar components (thin/thick disks and stellar halo)
    dfThinDisk    = agama.DistributionFunction(potential=model.potential, **iniDFThinDisk)
    dfThickDisk   = agama.DistributionFunction(potential=model.potential, **iniDFThickDisk)
    dfStellarHalo = agama.DistributionFunction(potential=model.potential, **iniDFStellarHalo)
    # composite DF of all stellar components except the bulge
    dfStellar     = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo)
    # composite DF of all stellar components including the bulge
    dfStellarAll  = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo, dfBulge)

    # replace the disk, halo and bulge SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfStellar, disklike=True, **iniSCMDisk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
    model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

    # we can compute the masses even though we don't know the density profile yet
    print("Masses of DF components: " \
        "Mdisk=%g Msun (Mthin=%g, Mthick=%g, Mstel.halo=%g); Mbulge=%g Msun; Mdarkhalo=%g Msun" % \
        (dfStellar.totalMass(), dfThinDisk.totalMass(), dfThickDisk.totalMass(), \
        dfStellarHalo.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

    # do a few more iterations to obtain the self-consistent density profile for the entire system
    for iteration in range(5):
        print("\033[1;37mStarting iteration #{:d}\033[0m".format(iteration+1))
        model.iterate()
        printoutInfo(model, "iter {:d}".format(iteration+1))

    # Sample 
    xv = agama.GalaxyModel(potential=model.potential, df=dfStellar, af=model.af).sample(2000)[0]
    cgal = coord.Galactocentric(x=-1*xv[:,0]*u.kpc, y=xv[:,1]*u.kpc, z=xv[:,2]*u.kpc, v_x=-1*xv[:,3]*u.km/u.s, v_y=xv[:,4]*u.km/u.s, v_z=xv[:,5]*u.km/u.s)
    
    print(cgal)
    
    w0 = gd.PhaseSpacePosition(cgal.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0].value
    lz = orbit.angular_momentum()[2][0]
    
    print(np.sum((etot>-0.16) & (etot<-0.06)))
    print(np.sum(etot<0))
    
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) # & (t['FeH']<-1)
    
    ebins = np.linspace(-0.30,-0.02,200)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.plot(lz, etot, 'k.', ms=3, alpha=0.1, mew=0, zorder=0)
    plt.plot(t['Lz'], t['E_tot_pot1'], 'o', color='tab:orange', zorder=0, ms=1, alpha=0.1)
    
    plt.xlim(-6,6)
    plt.ylim(-0.3, -0.02)
    #plt.ylim(-0.18, -0.02)
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

    ## output various profiles (only for stellar components)
    #print("\033[1;33mComputing density profiles and velocity distribution\033[0m")
    #modelStars = agama.GalaxyModel(model.potential, dfStellar, model.af)

def initialize_adisk(Nrand=50000, seed=2572, ret=True, graph=False):
    """Sample a thick disk distribution function with agama"""
    
    iniPotenThickDisk = dict(
        type = 'Disk',
        surfaceDensity = 3.0e+8,
        scaleRadius = 2.1,
        scaleHeight = -0.7
        )
    
    iniDFThickDisk = dict(
        type = 'Exponential',
        mass = .95e10,
        Jr0 = 63,
        Jz0 = 20,
        Jphi0 = 300,
        pr = 0.13,
        pz = 0.05,
        addJden = 20,
        addJvel = 40)
    
    iniDFThickDisk = dict(
        type = 'Exponential',
        mass    = 1e10,
        ## Parameters with dimensions of action [kpc*km/s]
        # scale action setting the radial velocity dispersion (~ .5*<Vr^2>/kappa)
        Jr0     = 22,
        # scale action setting the disk thickness and the vertical velocity dispersion ~(Vc*h)
        Jz0     = 5,
        # scale action setting the disk radius (~Rd*Vc)
        Jphi0   = 550,
        # power that controls radial decrease of dispersions
        pr	= -.25,
        pz	= -.1,
        # additional contribution to the sum of actions that affects the density profile
        addJden = 10,
        # same for the part that affects the velocity dispersion profiles
        addJvel = 700)
        
    ### parameters of the bulge+disk+stellar halo component of self-consistent model
    iniSCMDisk = dict(
        RminCyl = 0.1,
        RmaxCyl = 30,
        sizeRadialCyl = 20,
        zminCyl = 0.05,
        zmaxCyl = 10,
        sizeVerticalCyl= 16,
        )
    
    ### parameters for the potential solvers for the entire self-consistent model
    iniSCM = dict(
        rminSph = 0.01,
        rmaxSph = 1000.,
        sizeRadialSph = 50,
        lmaxAngularSph = 4,
        RminCyl = 0.1,
        RmaxCyl = 50,
        sizeRadialCyl = 30,
        zminCyl = 0.04,
        zmaxCyl = 20,
        sizeVerticalCyl = 30,
        useActionInterpolation = False
        )
    
    # define external unit system describing the data (including the ini parameters)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun
    
    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityThickDisk = agama.Density(**iniPotenThickDisk)

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityThickDisk, disklike=True))
    
    # compute the initial potential
    model.iterate()
    printoutInfo(model, 'init')
    
    Gpot=agama.GalaPotential(ham.potential)
    
    dfThickDisk = agama.DistributionFunction(potential=model.potential, **iniDFThickDisk)
    model.components[0] = agama.Component(df=dfThickDisk, disklike=True, **iniSCMDisk)
    
    for iteration in range(1,6):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, 'iter {:d}'.format(iteration))
    
    xv = agama.GalaxyModel(potential=model.potential, df=dfThickDisk, af=model.af).sample(Nrand*2)[0]
    
    cgal = coord.Galactocentric(x=-1*xv[:,0]*u.kpc, y=xv[:,1]*u.kpc, z=xv[:,2]*u.kpc, v_x=-1*xv[:,3]*u.km/u.s, v_y=xv[:,4]*u.km/u.s, v_z=xv[:,5]*u.km/u.s)
    
    ## select stars with high energy
    #w0 = gd.PhaseSpacePosition(cgal.transform_to(gc_frame).cartesian)
    #orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    #etot = orbit.energy()[0]
    #ind_highe = etot>-0.18*u.kpc**2*u.Myr**-2
    #cgal = cgal[ind_highe]
    
    ## pick randomly Nrand stars
    #np.random.seed(seed)
    #Ntot = np.size(cgal.x)
    #print('highe', Ntot)
    #ind = np.arange(0, Ntot, 1, dtype=int)
    #if Nrand<Ntot:
        #ind_rand = np.random.choice(ind, size=Nrand, replace=False)
    #else:
        #ind_rand = ind
    #c = cgal[ind_rand]
    
    w0 = gd.PhaseSpacePosition(cgal.transform_to(gc_frame).cartesian)
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
        #plt.ylim(-0.18, -0.02)
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
        plt.savefig('../plots/initialize_adisk_phase.png')
    
    if ret:
        ind = etot<0
        return c[ind]


####################
# Orbit integrations

def evolve_gala():
    """Evolve a set of disk particles in gala
    Energy and angular momentum conserved in a static, axisymmetric gravitational potential"""
    
    ic_list = initialize_disk(gala=True)
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2000)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    plt.plot(orbit.pos.x[0,:], orbit.pos.y[0,:], 'ko', ms=2, label='Initial')
    plt.plot(orbit.pos.x[-1,:], orbit.pos.y[-1,:], 'ro', ms=2, label='Final')
    
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.sca(ax[1])
    plt.plot(orbit.cylindrical.rho[0,:], orbit.cylindrical.z[0,:], 'ko', ms=2, label='Initial')
    plt.plot(orbit.cylindrical.rho[-1,:], orbit.cylindrical.z[-1,:], 'ro', ms=2, label='Final')
    
    plt.xlabel('R [kpc]')
    plt.ylabel('z [kpc]')
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()

def evolve_gala_elz():
    """Evolve a set of disk particles in gala
    Energy and angular momentum conserved in a static, axisymmetric gravitational potential"""
    
    ic_list = initialize_disk(gala=True)
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2000)
    
    energy = orbit.energy()
    l = orbit.angular_momentum()
    #print(np.shape(energy), np.shape(l))
    
    plt.close()
    plt.figure()
    
    plt.plot(l[2,0,:], energy[0,:], 'ko', ms=2, label='Initial')
    plt.plot(l[2,-1,:], energy[-1,:], 'ro', ms=2, label='Final')
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.legend()
    
    plt.tight_layout()

def evolve_gala_dg(dg='lmc', Nr=10, logspace=False, Nback=3000):
    """"""
    
    ic_list = initialize_disk(gala=True, Nr=Nr, const_arc=False, logspace=logspace)
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w1_disk = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    if dg=='sgr':
        dg_pot = gp.PlummerPotential(m=2e10*u.Msun, b=1*u.kpc, units=galactic)
        c_dg = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=27*u.kpc, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    else:
        dg_pot = gp.PlummerPotential(m=2e11*u.Msun, b=1*u.kpc, units=galactic)
        c_dg = coord.ICRS(ra=78.76*u.deg, dec=-69.19*u.deg, distance=10**(0.2*18.50+1)*u.pc, radial_velocity=262.2*u.km/u.s, pm_ra_cosdec=1.91*u.mas/u.yr, pm_dec=0.229*u.mas/u.yr)
        dg = 'lmc'
    
    w1_dg = gd.PhaseSpacePosition(c_dg.transform_to(gc_frame).cartesian)
    
    #Nback = 10000
    #Nback = 3000
    dt = 0.5*u.Myr
    # find initial LMC position
    dg_orbit_back = ham.integrate_orbit(w1_dg, dt=-dt, n_steps=Nback)
    w0_dg = dg_orbit_back[-1]
    
    # to match the static frame of lmc
    disk_orbit_back = ham.integrate_orbit(w1_disk, dt=-dt, n_steps=0)
    w0_disk = disk_orbit_back[-1]
    
    # integrate forward
    w0 = gd.combine((w0_dg, w0_disk))
    particle_pot = [None] * (N + 1)
    particle_pot[0] = dg_pot
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    orbit = nbody.integrate_orbit(dt=dt, n_steps=Nback)
    
    fname = '../data/logcircular_plummer_{:s}_{:05d}.h5'.format(dg, Nback)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::50,:]
    orbit_out.to_hdf5(fout)
    fout.close()
    
    energy = orbit.energy()
    l = orbit.angular_momentum()
    
    plt.close()
    plt.figure()
    
    plt.hist(energy.value[0,1:], bins=1000, histtype='step', color='k', label='Initial', density=True)
    plt.hist(energy.value[-1,1:], bins=1000, histtype='step', color='r', label='Final', density=True)
    
    plt.legend()
    
    plt.tight_layout()


def evolve_sgr(d=27*u.kpc, m=1.4e10*u.Msun, a=7*u.kpc, Napo=5, Nr=10, logspace=False, const_arc=True, sigma_z=0*u.kpc, sigma_vz=0*u.km/u.s, Nskip=100):
    """
    Initialize at apocenter
    d - Sgr heliocentric distance (default 27 kpc, reasonable range: 24-28 kpc, Vasiliev)
    m - Sgr Hernquist mass (default 1.4e10 Msun - H2, H1: 0.8e10 Msun, Laporte)
    a - Sgr Hernquist scale radius (default 7 kpc - H2, alternatives H2 13kpc, H1 8,16kpc, Laporte)
    Napo - number of apocenters to simulate
    Nr - number of radial orbits in the disk
    logspace - log spacing of radial orbits
    Nskip - number of timesteps to skip when saving orbits
    """
    
    # initialize Sgr
    sgr_pot = gp.HernquistPotential(m=m, c=a, units=galactic)
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    
    w1_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    # find initial Sgr position
    dt = 0.5*u.Myr
    Tback = 10*u.Gyr
    Nback = int((Tback/dt).decompose())
    
    sgr_orbit_back = ham.integrate_orbit(w1_sgr, dt=-dt, n_steps=Nback)
    
    # extract position at a given apocenter
    apo, tapo = sgr_orbit_back.apocenter(func=None, return_times=True)
    ind_apo = np.argmin(np.abs(sgr_orbit_back.t - tapo[-Napo]))
    w0_sgr = sgr_orbit_back[ind_apo]
    
    # calculate number of time steps to integrate forward
    ind = np.arange(Nback+1, dtype=int)
    Nfwd = ind[ind_apo]

    # initialize disk
    ic_list = initialize_disk(gala=True, Nr=Nr, const_arc=const_arc, logspace=logspace, sigma_z=sigma_z, sigma_vz=sigma_vz, verbose=False)
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w1_disk = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    # to match the static frame of sgr
    disk_orbit_back = ham.integrate_orbit(w1_disk, dt=-dt, n_steps=0)
    w0_disk = disk_orbit_back[-1]
    
    # integrate forward
    w0 = gd.combine((w0_sgr, w0_disk))
    particle_pot = [None] * (N + 1)
    particle_pot[0] = sgr_pot
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    orbit = nbody.integrate_orbit(dt=dt, n_steps=Nfwd)
    
    # save orbits
    if logspace:
        disk_label = 'logdisk'
    else:
        disk_label = 'disk'
    if const_arc:
        disk_label += '_flat'
    
    fname = '../data/sgr_hernquist_d.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}.h5'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr)
    #fname = '../data/sgr_hernquist_d.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}.h5'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::Nskip,:]
    orbit_out.to_hdf5(fout)
    fout.close()

def evolve_sgr_gse(d=27*u.kpc, m=1.4e10*u.Msun, a=1*u.kpc, Napo=5, Nskip=8, iskip=0, snap_skip=100):
    """
    Initialize at apocenter
    d - Sgr heliocentric distance (default 27 kpc, reasonable range: 24-28 kpc, Vasiliev)
    m - Sgr Hernquist mass (default 1.4e10 Msun - H2, H1: 0.8e10 Msun, Laporte)
    a - Sgr Hernquist scale radius (default 7 kpc - H2, alternatives H2 13kpc, H1 8,16kpc, Laporte)
    Napo - number of apocenters to simulate
    Nr - number of radial orbits in the disk
    logspace - log spacing of radial orbits
    Nskip - number of timesteps to skip when saving orbits
    """
    
    # initialize Sgr
    sgr_pot = gp.HernquistPotential(m=m, c=a, units=galactic)
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    
    w1_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    # find initial Sgr position
    dt = 0.5*u.Myr
    Tback = 10*u.Gyr
    Nback = int((Tback/dt).decompose())
    
    sgr_orbit_back = ham.integrate_orbit(w1_sgr, dt=-dt, n_steps=Nback)
    
    # extract position at a given apocenter
    apo, tapo = sgr_orbit_back.apocenter(func=None, return_times=True)
    ind_apo = np.argmin(np.abs(sgr_orbit_back.t - tapo[-Napo]))
    w0_sgr = sgr_orbit_back[ind_apo]
    
    # calculate number of time steps to integrate forward
    ind = np.arange(Nback+1, dtype=int)
    Nfwd = ind[ind_apo]

    # initialize disk
    #ic_list = initialize_halo(Nr=400)
    ic_list = initialize_gse(nskip=Nskip, iskip=iskip)
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w1_mw = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    # to match the static frame of sgr
    mw_orbit_back = ham.integrate_orbit(w1_mw, dt=-dt, n_steps=0)
    w0_mw = mw_orbit_back[-1]
    
    # integrate forward
    w0 = gd.combine((w0_sgr, w0_mw))
    particle_pot = [None] * (N + 1)
    particle_pot[0] = sgr_pot
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    orbit = nbody.integrate_orbit(dt=dt, n_steps=Nfwd)
    
    # save orbits
    #mw_label = 'halo'
    mw_label = 'gse'
    
    fname = '../data/sgr_hernquist_d.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_{:d}.{:d}.h5'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nskip, iskip)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::snap_skip,:]
    orbit_out.to_hdf5(fout)
    fout.close()

def evolve_sgr_stars(d=27*u.kpc, m=1.4e10*u.Msun, a=1*u.kpc, Napo=5, mw_label='halo', Nrand=50000, seed=3928, Nskip=8, iskip=0, snap_skip=100, test=False):
    """
    Initialize at apocenter
    d - Sgr heliocentric distance (default 27 kpc, reasonable range: 24-28 kpc, Vasiliev)
    m - Sgr Hernquist mass (default 1.4e10 Msun - H2, H1: 0.8e10 Msun, Laporte)
    a - Sgr Hernquist scale radius (default 7 kpc - H2, alternatives H2 13kpc, H1 8,16kpc, Laporte)
    Napo - number of apocenters to simulate
    Nr - number of radial orbits in the disk
    logspace - log spacing of radial orbits
    Nskip - number of timesteps to skip when saving orbits
    """
    
    # initialize Sgr
    sgr_pot = gp.HernquistPotential(m=m, c=a, units=galactic)
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    
    w1_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    # find initial Sgr position
    dt = 0.5*u.Myr
    Tback = 10*u.Gyr
    Nback = int((Tback/dt).decompose())
    
    sgr_orbit_back = ham.integrate_orbit(w1_sgr, dt=-dt, n_steps=Nback)
    
    # extract position at a given apocenter
    apo, tapo = sgr_orbit_back.apocenter(func=None, return_times=True)
    ind_apo = np.argmin(np.abs(sgr_orbit_back.t - tapo[-Napo]))
    w0_sgr = sgr_orbit_back[ind_apo]
    
    # calculate number of time steps to integrate forward
    ind = np.arange(Nback+1, dtype=int)
    Nfwd = ind[ind_apo]

    # initialize star particles
    if mw_label=='halo':
        c = initialize_halo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)[iskip::Nskip]
    elif mw_label=='ndisk':
        c = initialize_ndisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='nhalo':
        c = initialize_nhalo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='adisk':
        c = initialize_adisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    else:
        mw_label = 'td'
        c = initialize_td(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    w1_mw = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    if test:
        print(c.x[0], np.size(c), tapo[-Napo], Nfwd)
        return 0
    
    # to match the static frame of sgr
    mw_orbit_back = ham.integrate_orbit(w1_mw, dt=-dt, n_steps=0)
    w0_mw = mw_orbit_back[-1]
    
    # integrate forward
    w0 = gd.combine((w0_sgr, w0_mw))
    particle_pot = [None] * (N + 1)
    particle_pot[0] = sgr_pot
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    orbit = nbody.integrate_orbit(dt=dt, n_steps=Nfwd)
    
    # save orbits
    fname = '../data/sgr_hernquist_d.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip, iskip)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::snap_skip,:]
    orbit_out.to_hdf5(fout)
    fout.close()


def interact_sgr_stars(d=27*u.kpc, m=1.4e10*u.Msun, fm=16, a=1*u.kpc, fa=5, Napo=5, mw_label='halo', Nrand=50000, seed=3928, Nskip=1, iskip=0, snap_skip=100, df=True, test=False, graph=True, verbose=True):
    """
    Initialize at apocenter
    d - Sgr heliocentric distance (default 27 kpc, reasonable range: 24-28 kpc, Vasiliev)
    m - Sgr Hernquist mass (default 1.4e10 Msun - H2, H1: 0.8e10 Msun, Laporte)
    a - Sgr Hernquist scale radius (default 7 kpc - H2, alternatives H2 13kpc, H1 8,16kpc, Laporte)
    """
    
    # initialize Sgr location
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    cg_sgr = c_sgr.transform_to(coord.Galactocentric())
    xp = np.array([cg_sgr.x.to(u.kpc).value, cg_sgr.y.to(u.kpc).value, cg_sgr.z.to(u.kpc).value]) * u.kpc
    vp = np.array([cg_sgr.v_x.to(u.km/u.s).value, cg_sgr.v_y.to(u.km/u.s).value, cg_sgr.v_z.to(u.km/u.s).value]) * u.km/u.s

    # Hernquist profile
    potential_perturb = 2
    par_perturb = np.array([m.si.value, a.si.value, 0., 0., 0.])
    
    #w1_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    ## find initial Sgr position
    #dt = 0.5*u.Myr
    #Tback = 10*u.Gyr
    #Nback = int((Tback/dt).decompose())
    #Napo = 5
    
    #sgr_orbit_back = ham.integrate_orbit(w1_sgr, dt=-dt, n_steps=Nback)
    
    ## extract position at a given apocenter
    #apo, tapo = sgr_orbit_back.apocenter(func=None, return_times=True)
    #ind_apo = np.argmin(np.abs(sgr_orbit_back.t - tapo[-Napo]))
    #w0_sgr = sgr_orbit_back[ind_apo]
    ##print("density", ham.potential['halo'].density(w1_sgr).si)
    
    ## calculate number of time steps to integrate forward
    #ind = np.arange(Nback+1, dtype=int)
    #Nfwd = ind[ind_apo]
    ##print(tapo[-Napo], Nfwd)

    # initialize star particles
    if mw_label=='halo':
        c = initialize_halo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)[iskip::Nskip]
    elif mw_label=='ndisk':
        c = initialize_ndisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='nhalo':
        c = initialize_nhalo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='adisk':
        c = initialize_adisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    else:
        mw_label = 'td'
        c = initialize_td(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    
    cg = c.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0].value
    lz = orbit.angular_momentum()[2][0]
    
    if test:
        print(xp, vp)
        print(np.size(cg.x))
        return 0
    
    
    ##MilkyWayPotential
    #Mh = 5.4e11*u.Msun
    #Rh = 15.62*u.kpc
    #Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
    #par_gal = [5e9*u.Msun, 1*u.kpc, 6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, Vh, Rh, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)]
    
    #MilkyWayPotential2022
    Mh = 554271110519.767*u.Msun
    Rh = 15.625903558190732*u.kpc
    Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
    par_gal = [5e9*u.Msun, 1*u.kpc, 47716664286.85036*u.Msun, 3*u.kpc, 0.26*u.kpc, Vh, Rh, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)]
    
    potential = 6
    par_pot = np.array([x_.si.value for x_ in par_gal])
    
    # leapfrog integrator
    integrator = 0
    
    # trial orbit to find total simulation time (4 pericenters or 300kpc)
    # orbit integration times
    dt = 0.5*u.Myr
    T = 8*u.Gyr
    direction = int(-1)
    t = np.arange(0, direction*(T+dt).to(u.Myr).value, direction*dt.to(u.Myr).value)*u.Myr
    
    if df:
        x1, x2, x3, v1, v2, v3, mass = interact.df_orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, T.to(u.s).value, dt.to(u.s).value, direction, m.to(u.kg).value, a.to(u.m).value, fm)
        mass = mass*u.kg
    else:
        x1, x2, x3, v1, v2, v3 = interact.orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, integrator, T.to(u.s).value, dt.to(u.s).value, direction)
    
    co = coord.Galactocentric(x=(x1*u.m).to(u.kpc), y=(x2*u.m).to(u.kpc), z=(x3*u.m).to(u.kpc), v_x=(v1*u.m/u.s).to(u.km/u.s), v_y=(v2*u.m/u.s).to(u.km/u.s), v_z=(v3*u.m/u.s).to(u.km/u.s))
    
    rgal = co.spherical.distance
    ind_apo, _ = find_peaks(rgal)
    ind_peri, _ = find_peaks(-rgal)
    
    if np.size(ind_apo)>=4:
        Nstart = ind_apo[3]
        Nstart = ind_apo[Napo]
    else:
        Nstart = np.argmin(np.abs(rgal - 300*u.kpc))
    
    # simulation time
    dt = 0.5*u.Myr
    T = np.abs(t[Nstart])
    #T = 510*u.Myr
    print(T)
    
    root = 'interact{:d}_d.{:.1f}_m.{:.2f}.{:.2f}_a.{:02.1f}.{:.1f}_{:s}_N.{:06d}'.format(df, d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, fm, a.to(u.kpc).value, fa, mw_label, Nrand)
    fname = '../data/snaps_{:s}.h5'.format(root)
    Nskip = 200
    
    t1 = time.time()
    if df:
        x1, x2, x3, v1, v2, v3 = interact.df_interact(par_perturb, m.si.value, fm, fa*a.si.value, xp.to(u.m).value, vp.to(u.m/u.s).value, T.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, cg.x.to(u.m).value, cg.y.to(u.m).value, cg.z.to(u.m).value, cg.v_x.to(u.m/u.s).value, cg.v_y.to(u.m/u.s).value, cg.v_z.to(u.m/u.s).value, fname, Nskip)
    else:
        Tenc = 0*u.Gyr
        x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xp.to(u.m).value, vp.to(u.m/u.s).value, Tenc.to(u.s).value, T.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, cg.x.to(u.m).value, cg.y.to(u.m).value, cg.z.to(u.m).value, cg.v_x.to(u.m/u.s).value, cg.v_y.to(u.m/u.s).value, cg.v_z.to(u.m/u.s).value, fname, Nskip)
    t2 = time.time()
    
    if verbose:
        print(t2 - t1)
    
    stars = {}
    stars['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stars['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    cg_ = coord.Galactocentric(x=stars['x'][0], y=stars['x'][1], z=stars['x'][2], v_x=stars['v'][0], v_y=stars['v'][1], v_z=stars['v'][2])
    w0_ = gd.PhaseSpacePosition(cg_.cartesian)
    orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
    etot_ = orbit_.energy()[0].value
    lz_ = orbit_.angular_momentum()[2][0]
    
    # save
    outdict = dict(cg=cg_, etot=etot_, lz=lz_)
    #root = 'interact{:d}_d.{:.1f}_m.{:.2f}.{:.2f}_a.{:02.1f}.{:.1f}_{:s}_N.{:06d}'.format(df, d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, fm, a.to(u.kpc).value, fa, mw_label, Nrand)
    pickle.dump(outdict, open('../data/models/model_{:s}.pkl'.format(root), 'wb'))
    
    # Sgr orbit
    direction = int(-1)
    t = np.arange(0, direction*(T+dt).to(u.Myr).value, direction*dt.to(u.Myr).value)*u.Myr

    if df:
        x1, x2, x3, v1, v2, v3, mass = interact.df_orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, T.to(u.s).value, dt.to(u.s).value, direction, m.to(u.kg).value, a.to(u.m).value, fm)
        mass = (mass*u.kg).to(u.Msun)
    else:
        x1, x2, x3, v1, v2, v3 = interact.orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, integrator, T.to(u.s).value, dt.to(u.s).value, direction)
        mass = m
    
    co = coord.Galactocentric(x=(x1*u.m).to(u.kpc), y=(x2*u.m).to(u.kpc), z=(x3*u.m).to(u.kpc), v_x=(v1*u.m/u.s).to(u.km/u.s), v_y=(v2*u.m/u.s).to(u.km/u.s), v_z=(v3*u.m/u.s).to(u.km/u.s))
    
    sgr_orbit = dict(c=co, t=t, m=m)
    pickle.dump(sgr_orbit, open('../data/sgr_orbit_{:s}.pkl'.format(root), 'wb'))
    
    if graph:
        # setup plotting
        if 'halo' in mw_label:
            elz_xlim = [-5,5]
            elz_ylim = [-0.18, -0.04]
            rvr_xlim = [1,100]
            rvr_ylim = [-400,400]
            ind_model = (outdict['lz']<0) & (np.abs(outdict['cg'].z)>2*u.kpc) & (outdict['cg'].spherical.distance<20*u.kpc) & (outdict['cg'].spherical.distance>5*u.kpc)
    
        elif 'disk' in mw_label:
            elz_xlim = [-6,2]
            elz_ylim = [-0.18, -0.04]
            rvr_xlim = [1,30]
            rvr_ylim = [-300,300]
            ind_model = (np.abs(outdict['cg'].z)>2*u.kpc) & (outdict['cg'].spherical.distance<15*u.kpc) & (outdict['cg'].spherical.distance>5*u.kpc)
        else:
            elz_xlim = [-6,6]
            elz_ylim = [-0.18, -0.02]
            rvr_xlim = [1,100]
            rvr_ylim = [-400,400]
            ind_model = np.ones(len(outdict['lz']), dtype=bool)
        
        # plot phase-space diagnostics
        plt.close()
        fig, ax = plt.subplots(1,4,figsize=(18,4.5))
        
        # E-Lz
        plt.sca(ax[0])
        zoff = 10*u.kpc
        plt.plot(cg.x, cg.z + zoff, 'o', color='0.5', ms=0.5, mew=0, alpha=0.1)
        plt.plot(cg_.x, cg_.z - zoff, 'ko', ms=0.5, mew=0, alpha=0.1)
        plt.xlim(-22,22)
        plt.ylim(-22,22)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlabel('x [kpc]')
        plt.ylabel('z [kpc]')
        
        plt.sca(ax[1])
        plt.plot(t.to(u.Gyr), co.spherical.distance, 'k-')
        plt.xlabel('Time [Gyr]')
        plt.ylabel('r$_{gal}$ [kpc]')
        plt.text(0.9, 0.9, '{:.3g}'.format(np.max(mass)), transform=plt.gca().transAxes, va='top', ha='right', fontsize='small')
        
        plt.sca(ax[2])
        plt.plot(lz_, etot_, 'b.', ms=0.1, mew=0, alpha=0.01)
        plt.plot(lz_[ind_model], etot_[ind_model], 'ko', ms=1, mew=0, alpha=0.1)
        eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
        for k, e in enumerate(eridge):
            plt.axhline(e, lw=0.5, alpha=0.5)
        plt.xlim(-6,6)
        plt.ylim(-0.18, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        
        plt.sca(ax[3])
        ebins = np.linspace(-0.18,-0.07,100)
        plt.hist(etot_[ind_model], bins=ebins, color='k', density=True, histtype='step')
        plt.hist(etot[ind_model], bins=ebins, color='k', density=True, histtype='step', alpha=0.2)
        for k, e in enumerate(eridge):
            plt.axvline(e, lw=0.5, alpha=0.5)
        
        plt.xlim(-0.16, -0.09)
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
        
        plt.tight_layout()
        plt.savefig('../plots/phasespace_{:s}.png'.format(root))


def interact_lmc_stars(d=27*u.kpc, m=2.5e11*u.Msun, fm=16, a=5*u.kpc, fa=5, Napo=5, mw_label='halo', Nrand=50000, seed=3928, Nskip=1, iskip=0, snap_skip=100, df=True, test=False, graph=True, verbose=True):
    """
    Initialize at apocenter
    d - Sgr heliocentric distance (default 27 kpc, reasonable range: 24-28 kpc, Vasiliev)
    m - Sgr Hernquist mass (default 1.4e10 Msun - H2, H1: 0.8e10 Msun, Laporte)
    a - Sgr Hernquist scale radius (default 7 kpc - H2, alternatives H2 13kpc, H1 8,16kpc, Laporte)
    """
    
    # initialize Sgr location
    dg = 'lmc'
    c_dg = coord.ICRS(ra=78.76*u.deg, dec=-69.19*u.deg, distance=10**(0.2*18.50+1)*u.pc, radial_velocity=262.2*u.km/u.s, pm_ra_cosdec=1.91*u.mas/u.yr, pm_dec=0.229*u.mas/u.yr)
    #c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    cg_dg = c_dg.transform_to(coord.Galactocentric())
    xp = np.array([cg_dg.x.to(u.kpc).value, cg_dg.y.to(u.kpc).value, cg_dg.z.to(u.kpc).value]) * u.kpc
    vp = np.array([cg_dg.v_x.to(u.km/u.s).value, cg_dg.v_y.to(u.km/u.s).value, cg_dg.v_z.to(u.km/u.s).value]) * u.km/u.s

    # Hernquist profile
    potential_perturb = 2
    par_perturb = np.array([m.si.value, a.si.value, 0., 0., 0.])
    
    #w1_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    ## find initial Sgr position
    #dt = 0.5*u.Myr
    #Tback = 10*u.Gyr
    #Nback = int((Tback/dt).decompose())
    #Napo = 5
    
    #sgr_orbit_back = ham.integrate_orbit(w1_sgr, dt=-dt, n_steps=Nback)
    
    ## extract position at a given apocenter
    #apo, tapo = sgr_orbit_back.apocenter(func=None, return_times=True)
    #ind_apo = np.argmin(np.abs(sgr_orbit_back.t - tapo[-Napo]))
    #w0_sgr = sgr_orbit_back[ind_apo]
    ##print("density", ham.potential['halo'].density(w1_sgr).si)
    
    ## calculate number of time steps to integrate forward
    #ind = np.arange(Nback+1, dtype=int)
    #Nfwd = ind[ind_apo]
    ##print(tapo[-Napo], Nfwd)

    # initialize star particles
    if mw_label=='halo':
        c = initialize_halo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)[iskip::Nskip]
    elif mw_label=='ndisk':
        c = initialize_ndisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='nhalo':
        c = initialize_nhalo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='adisk':
        c = initialize_adisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    else:
        mw_label = 'td'
        c = initialize_td(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    
    cg = c.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0].value
    lz = orbit.angular_momentum()[2][0]
    
    if test:
        print(xp, vp)
        print(np.size(cg.x))
        return 0
    
    
    ##MilkyWayPotential
    #Mh = 5.4e11*u.Msun
    #Rh = 15.62*u.kpc
    #Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
    #par_gal = [5e9*u.Msun, 1*u.kpc, 6.8e10*u.Msun, 3*u.kpc, 0.28*u.kpc, Vh, Rh, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)]
    
    #MilkyWayPotential2022
    Mh = 554271110519.767*u.Msun
    Rh = 15.625903558190732*u.kpc
    Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
    par_gal = [5e9*u.Msun, 1*u.kpc, 47716664286.85036*u.Msun, 3*u.kpc, 0.26*u.kpc, Vh, Rh, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)]
    
    potential = 6
    par_pot = np.array([x_.si.value for x_ in par_gal])
    
    # leapfrog integrator
    integrator = 0
    
    # trial orbit to find total simulation time (4 pericenters or 300kpc)
    # orbit integration times
    dt = 0.5*u.Myr
    T = 8*u.Gyr
    direction = int(-1)
    t = np.arange(0, direction*(T+dt).to(u.Myr).value, direction*dt.to(u.Myr).value)*u.Myr
    
    if df:
        x1, x2, x3, v1, v2, v3, mass = interact.df_orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, T.to(u.s).value, dt.to(u.s).value, direction, m.to(u.kg).value, a.to(u.m).value, fm)
        mass = mass*u.kg
    else:
        x1, x2, x3, v1, v2, v3 = interact.orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, integrator, T.to(u.s).value, dt.to(u.s).value, direction)
    
    co = coord.Galactocentric(x=(x1*u.m).to(u.kpc), y=(x2*u.m).to(u.kpc), z=(x3*u.m).to(u.kpc), v_x=(v1*u.m/u.s).to(u.km/u.s), v_y=(v2*u.m/u.s).to(u.km/u.s), v_z=(v3*u.m/u.s).to(u.km/u.s))
    
    rgal = co.spherical.distance
    ind_apo, _ = find_peaks(rgal)
    ind_peri, _ = find_peaks(-rgal)
    
    if dg=='lmc':
        Nstart = np.argmin(np.abs(rgal - 300*u.kpc))
    else:
        if np.size(ind_apo)>=4:
            Nstart = ind_apo[3]
        else:
            Nstart = np.argmin(np.abs(rgal - 300*u.kpc))
    
    # simulation time
    dt = 0.5*u.Myr
    T = np.abs(t[Nstart])
    #T = 510*u.Myr
    print(T)
    
    root = 'interact{:d}_{:s}_d.{:.1f}_m.{:.2f}.{:.2f}_a.{:02.1f}.{:.1f}_{:s}_N.{:06d}'.format(df, dg, d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, fm, a.to(u.kpc).value, fa, mw_label, Nrand)
    fname = '../data/snaps_{:s}.h5'.format(root)
    Nskip = 200
    
    t1 = time.time()
    if df:
        x1, x2, x3, v1, v2, v3 = interact.df_interact(par_perturb, m.si.value, fm, fa*a.si.value, xp.to(u.m).value, vp.to(u.m/u.s).value, T.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, cg.x.to(u.m).value, cg.y.to(u.m).value, cg.z.to(u.m).value, cg.v_x.to(u.m/u.s).value, cg.v_y.to(u.m/u.s).value, cg.v_z.to(u.m/u.s).value, fname, Nskip)
    else:
        Tenc = 0*u.Gyr
        x1, x2, x3, v1, v2, v3 = interact.general_interact(par_perturb, xp.to(u.m).value, vp.to(u.m/u.s).value, Tenc.to(u.s).value, T.to(u.s).value, dt.to(u.s).value, par_pot, potential, potential_perturb, cg.x.to(u.m).value, cg.y.to(u.m).value, cg.z.to(u.m).value, cg.v_x.to(u.m/u.s).value, cg.v_y.to(u.m/u.s).value, cg.v_z.to(u.m/u.s).value, fname, Nskip)
    t2 = time.time()
    
    if verbose:
        print(t2 - t1)
    
    stars = {}
    stars['x'] = (np.array([x1, x2, x3])*u.m).to(u.kpc)
    stars['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    cg_ = coord.Galactocentric(x=stars['x'][0], y=stars['x'][1], z=stars['x'][2], v_x=stars['v'][0], v_y=stars['v'][1], v_z=stars['v'][2])
    w0_ = gd.PhaseSpacePosition(cg_.cartesian)
    orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
    etot_ = orbit_.energy()[0].value
    lz_ = orbit_.angular_momentum()[2][0]
    
    # save
    outdict = dict(cg=cg_, etot=etot_, lz=lz_)
    #root = 'interact{:d}_d.{:.1f}_m.{:.2f}.{:.2f}_a.{:02.1f}.{:.1f}_{:s}_N.{:06d}'.format(df, d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, fm, a.to(u.kpc).value, fa, mw_label, Nrand)
    pickle.dump(outdict, open('../data/models/model_{:s}.pkl'.format(root), 'wb'))
    
    # Sgr orbit
    direction = int(-1)
    t = np.arange(0, direction*(T+dt).to(u.Myr).value, direction*dt.to(u.Myr).value)*u.Myr

    if df:
        x1, x2, x3, v1, v2, v3, mass = interact.df_orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, T.to(u.s).value, dt.to(u.s).value, direction, m.to(u.kg).value, a.to(u.m).value, fm)
        mass = (mass*u.kg).to(u.Msun)
    else:
        x1, x2, x3, v1, v2, v3 = interact.orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, integrator, T.to(u.s).value, dt.to(u.s).value, direction)
        mass = m
    
    co = coord.Galactocentric(x=(x1*u.m).to(u.kpc), y=(x2*u.m).to(u.kpc), z=(x3*u.m).to(u.kpc), v_x=(v1*u.m/u.s).to(u.km/u.s), v_y=(v2*u.m/u.s).to(u.km/u.s), v_z=(v3*u.m/u.s).to(u.km/u.s))
    
    sgr_orbit = dict(c=co, t=t, m=m)
    pickle.dump(sgr_orbit, open('../data/sgr_orbit_{:s}.pkl'.format(root), 'wb'))
    
    if graph:
        # setup plotting
        if 'halo' in mw_label:
            elz_xlim = [-5,5]
            elz_ylim = [-0.18, -0.04]
            rvr_xlim = [1,100]
            rvr_ylim = [-400,400]
            ind_model = (outdict['lz']<0) & (np.abs(outdict['cg'].z)>2*u.kpc) & (outdict['cg'].spherical.distance<20*u.kpc) & (outdict['cg'].spherical.distance>5*u.kpc)
    
        elif 'disk' in mw_label:
            elz_xlim = [-6,2]
            elz_ylim = [-0.18, -0.04]
            rvr_xlim = [1,30]
            rvr_ylim = [-300,300]
            ind_model = (np.abs(outdict['cg'].z)>2*u.kpc) & (outdict['cg'].spherical.distance<15*u.kpc) & (outdict['cg'].spherical.distance>5*u.kpc)
        else:
            elz_xlim = [-6,6]
            elz_ylim = [-0.18, -0.02]
            rvr_xlim = [1,100]
            rvr_ylim = [-400,400]
            ind_model = np.ones(len(outdict['lz']), dtype=bool)
        
        # plot phase-space diagnostics
        plt.close()
        fig, ax = plt.subplots(1,4,figsize=(18,4.5))
        
        # E-Lz
        plt.sca(ax[0])
        zoff = 10*u.kpc
        plt.plot(cg.x, cg.z + zoff, 'o', color='0.5', ms=0.5, mew=0, alpha=0.1)
        plt.plot(cg_.x, cg_.z - zoff, 'ko', ms=0.5, mew=0, alpha=0.1)
        plt.xlim(-22,22)
        plt.ylim(-22,22)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlabel('x [kpc]')
        plt.ylabel('z [kpc]')
        
        plt.sca(ax[1])
        plt.plot(t.to(u.Gyr), co.spherical.distance, 'k-')
        plt.xlabel('Time [Gyr]')
        plt.ylabel('r$_{gal}$ [kpc]')
        plt.text(0.9, 0.9, '{:.3g}'.format(np.max(mass)), transform=plt.gca().transAxes, va='top', ha='right', fontsize='small')
        
        plt.sca(ax[2])
        plt.plot(lz_, etot_, 'b.', ms=0.1, mew=0, alpha=0.01)
        plt.plot(lz_[ind_model], etot_[ind_model], 'ko', ms=1, mew=0, alpha=0.1)
        eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
        for k, e in enumerate(eridge):
            plt.axhline(e, lw=0.5, alpha=0.5)
        plt.xlim(-6,6)
        plt.ylim(-0.18, -0.02)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        
        plt.sca(ax[3])
        ebins = np.linspace(-0.18,-0.07,100)
        plt.hist(etot_[ind_model], bins=ebins, color='k', density=True, histtype='step')
        plt.hist(etot[ind_model], bins=ebins, color='k', density=True, histtype='step', alpha=0.2)
        for k, e in enumerate(eridge):
            plt.axvline(e, lw=0.5, alpha=0.5)
        
        plt.xlim(-0.16, -0.09)
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
        
        plt.tight_layout()
        plt.savefig('../plots/phasespace_{:s}.png'.format(root))



# mass-loss

def massloss_history():
    """"""
    
    h = 0.7
    om = 0.3
    rvir = 163*h**-1 * om**(-1/3)
    print(rvir)
    
    N = 100
    x = np.zeros(N)*u.kpc
    y = np.zeros(N)*u.kpc
    z = np.linspace(0.1, 1.2*rvir, N)*u.kpc
    z = np.logspace(0, np.log10(1.2*rvir), N)*u.kpc
    c = coord.Galactocentric(x=x, y=y, z=z)
    
    q = np.array([x.value, y.value, z.value]).T
    
    menc = np.empty(N)*u.Msun
    for i in range(N):
        menc[i] = ham.potential.mass_enclosed(q[i])
    
    rg = c.spherical.distance
    m = np.logspace(8,11,10)*u.Msun
    rs = 10*u.kpc
    rt = rg * (m[:,np.newaxis]/(3*menc))**(1/3) / rs
    
    f = 0.01
    dm = f*m[:,np.newaxis]/rt
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(rg, rt.T, '-')
    
    plt.axvline(rvir)
    plt.axhline(1)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.ylabel('$r_t$ / $r_s$')
    
    plt.sca(ax[1])
    plt.plot(rg, dm.T, '-')
    
    plt.axvline(rvir)
    plt.gca().set_yscale('log')
    plt.ylabel('$\delta m$ [$M_\odot$/Myr]')
    plt.xlabel('$r_{gal}$ [kpc]')

    plt.tight_layout()

def sgr_minit(d=27*u.kpc):
    """"""
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    cg_sgr = c_sgr.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    dt = 1*u.Myr
    T = 3*u.Gyr
    nstep = int((T/dt).decompose())
    orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=nstep)
    
    q = np.array([orbit.x.value, orbit.y.value, orbit.z.value]).T
    N = np.size(orbit.x)
    
    menc = np.empty(N)*u.Msun
    for i in range(N):
        menc[i] = ham.potential.mass_enclosed(q[i])
    
    rg = orbit.spherical.distance
    m = np.logspace(8,11,4)*u.Msun
    rs = 10*u.kpc
    rt = rg * (m[:,np.newaxis]/(3*menc))**(1/3) / rs
    
    f = 1e-3
    dm = f*m[:,np.newaxis]/rt
    
    h = 0.7
    om = 0.3
    rvir = 163*h**-1 * om**(-1/3)
    
    m_inc = np.sum(dm, axis=1)
    print(m_inc, m, m+m_inc)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(rg, rt.T, '-')
    
    plt.axvline(rvir)
    plt.axhline(1)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.ylabel('$r_t$ / $r_s$')
    
    plt.sca(ax[1])
    plt.plot(rg, dm.T, '-')
    
    plt.axvline(rvir)
    plt.gca().set_yscale('log')
    plt.ylabel('$\delta m$ [$M_\odot$/Myr]')
    plt.xlabel('$r_{gal}$ [kpc]')

    plt.tight_layout()

def sgr_orbit(d=27*u.kpc, mi=1.4e10*u.Msun, f=5e-2, rs=1*u.kpc):
    """"""
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=d, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    cg_sgr = c_sgr.transform_to(coord.Galactocentric())
    w0 = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    xp = np.array([cg_sgr.x.to(u.kpc).value, cg_sgr.y.to(u.kpc).value, cg_sgr.z.to(u.kpc).value]) * u.kpc
    vp = np.array([cg_sgr.v_x.to(u.km/u.s).value, cg_sgr.v_y.to(u.km/u.s).value, cg_sgr.v_z.to(u.km/u.s).value]) * u.km/u.s
    
    # potential
    #MilkyWayPotential2022
    Mh = 554271110519.767*u.Msun
    Rh = 15.625903558190732*u.kpc
    Vh = np.sqrt(G*Mh/Rh).to(u.km/u.s)
    par_gal = [5e9*u.Msun, 1*u.kpc, 47716664286.85036*u.Msun, 3*u.kpc, 0.26*u.kpc, Vh, Rh, 0*u.rad, 1*u.Unit(1), 1*u.Unit(1), 1*u.Unit(1)]
    
    potential = 6
    par_pot = np.array([x_.si.value for x_ in par_gal])
    #print(ham.potential.parameters)
    
    # orbit integration times
    dt = 0.5*u.Myr
    T = 7*u.Gyr
    direction = int(-1)
    t = np.arange(0, direction*(T+dt).to(u.Myr).value, direction*dt.to(u.Myr).value)
    
    # Sgr structural properties
    #mi = 1e9*u.Msun
    #rs = 1*u.kpc
    #f = 5e-2
    #f = 5e-1
    #f = 5e-3
    
    x1, x2, x3, v1, v2, v3, mass = interact.df_orbit(xp.to(u.m).value, vp.to(u.m/u.s).value, par_pot, potential, T.to(u.s).value, dt.to(u.s).value, direction, mi.to(u.kg).value, rs.to(u.m).value, f)
    mass = (mass*u.kg).to(u.Msun)
    
    print('{:g}'.format(np.max(mass)))
    
    co = coord.Galactocentric(x=(x1*u.m).to(u.kpc), y=(x2*u.m).to(u.kpc), z=(x3*u.m).to(u.kpc), v_x=(v1*u.m/u.s).to(u.km/u.s), v_y=(v2*u.m/u.s).to(u.km/u.s), v_z=(v3*u.m/u.s).to(u.km/u.s))
    
    rgal = co.spherical.distance
    ind_apo, _ = find_peaks(rgal)
    ind_peri, _ = find_peaks(-rgal)
    
    if np.size(ind_apo)>=4:
        Nstart = ind_apo[3]
    else:
        Nstart = np.argmin(np.abs(rgal - 300*u.kpc))
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    plt.sca(ax[0])
    plt.plot(co.x, co.z, 'k-')
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.sca(ax[1])
    plt.plot(t, rgal, 'k-')
    plt.plot(t[ind_apo], rgal[ind_apo], 'bo')
    plt.plot(t[ind_peri], rgal[ind_peri], 'ro')
    plt.axvline(t[Nstart])
    
    plt.sca(ax[2])
    plt.plot(t, mass.to(u.Msun), 'k-')
    plt.gca().set_yscale('log')

    plt.tight_layout()


def batch_interact():
    """Run the C interaction code for a grid of input parameters"""
    
    #d = np.linspace(26.6,28.6,11) * u.kpc
    #fm = np.linspace(10,20,11)
    
    #fm = np.logspace(0,3,7)
    #print(fm)
    #d = np.linspace(26,29,7) * u.kpc
    #print(d)
    d = np.arange(27,28.1,0.5) * u.kpc
    fm = np.arange(0,1.1,0.1)
    
    d = np.array([26.5])*u.kpc
    fm = np.arange(0.5,1.01,0.05)
    
    print(np.size(d), np.size(fm), np.size(d)*np.size(fm)*3/60)
    
    for d_ in d:
        for fm_ in fm:
            print(d_, fm_)
            interact_sgr_stars(mw_label='ndisk', Nrand=100000, m=5e8*u.Msun, a=0.1*u.kpc, fa=50, d=d_, fm=fm_, graph=True, verbose=True)


#from colorio.cs import ColorCoordinates

def grid_cmap():
    """Develop a 2D colormap for plotting a grid of histograms"""
    
    x_ = np.linspace(0,1,100)
    xx_, yy_ = np.meshgrid(x_, x_)
    xx = np.ravel(xx_)
    yy = np.ravel(yy_)
    
    # corners chosen to have the same S and L in HSL, and to be perceptually most distant
    # v00 -- red, v10 -- blue, v01 -- orange, v11 -- teal
    hex_corners = ['#f00070', '#0029f0', '#f0e200', '#00f0ef']
    
    # create colorio RGB color coordinates
    corners = [ColorCoordinates(mpl.colors.to_rgb(x), 'srgb1') for x in hex_corners]
    
    # transfer corners to CIELAB color space
    for c in corners:
        c.convert('cielab')
    
    # create a list of colors based on their position in the xy plane using bilinear interpolation
    colors = []
    for i in range(np.size(xx)):
        cval = (1 - xx[i])*(1 - yy[i])*corners[0].data + xx[i]*(1 - yy[i])*corners[1].data + (1 - xx[i])*yy[i]*corners[2].data + xx[i]*yy[i]*corners[3].data
        c_ = ColorCoordinates(cval, 'cielab')
        c_.convert('srgbhex', mode='clip')
        colors += [np.array2string(c_.data)[1:-1]]
    
    plt.close()
    plt.figure()
    
    plt.scatter(xx, yy, c=colors, s=50)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../plots/grid_cmap.png')
    
def get_2dcolor(x, y):
    """"""
    hex_corners = ['#f00070', '#0029f0', '#f0e200', '#00f0ef']
    
    # create colorio RGB color coordinates
    corners = [ColorCoordinates(mpl.colors.to_rgb(x), 'srgb1') for x in hex_corners]
    
    # transfer corners to CIELAB color space
    for c in corners:
        c.convert('cielab')
    
    # bilinear interpolation to get the color based on its x,y position
    cval = (1 - x)*(1 - y)*corners[0].data + x*(1 - y)*corners[1].data + (1 - x)*y*corners[2].data + x*y*corners[3].data
    c_ = ColorCoordinates(cval, 'cielab')
    c_.convert('srgbhex', mode='clip')
    
    return np.array2string(c_.data)[1:-1]
    

def ehist_grid():
    """"""
    
    d_full = np.linspace(26.6,28.6,11) * u.kpc
    fm_full = np.linspace(10,20,11)
    mw_label = 'ndisk'
    Nrand = 50000
    m = 2e10*u.Msun
    a = 1*u.kpc
    fa = 5
    
    d_grid = d_full[::1]
    fm_grid = fm_full[::1]
    
    nrow = np.size(d_grid)
    ncol = np.size(fm_grid)
    
    xg = np.linspace(0,1,ncol)
    yg = np.linspace(1,0,nrow)
    xx_, yy_ = np.meshgrid(xg, yg)
    #xx = xx_.ravel()
    #yy = yy_.ravel()
    
    colors = []
    for x, y in zip(xx_.ravel(), yy_.ravel()):
        colors += [get_2dcolor(x, y)]
    
    ebins = np.linspace(-0.18,-0.07,100)
    eridge = np.array([-0.145, -0.134, -0.127, -0.122, -0.116])
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(20,20), sharex=True, sharey=True)
    
    for i, d in enumerate(d_grid):
        for j, fm in enumerate(fm_grid):
            ij = i*ncol + j
            root = 'interact_d.{:.1f}_m.{:.1f}.{:.1f}_a.{:02.0f}.{:.1f}_{:s}_N.{:06d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, fm, a.to(u.kpc).value, fa, mw_label, Nrand)
            mdisk = pickle.load(open('../data/models/model_{:s}.pkl'.format(root), 'rb'))
            ind_mdisk = (np.abs(mdisk['cg'].z)>2*u.kpc) & (mdisk['cg'].spherical.distance<15*u.kpc) & (mdisk['cg'].spherical.distance>5*u.kpc)

            plt.sca(ax[i,j])
            plt.hist(mdisk['etot'][ind_mdisk], color=colors[ij], bins=ebins, density=True, histtype='step', alpha=1)
            
            for k, e in enumerate(eridge):
                plt.axvline(e, lw=0.5, alpha=1, color='k', ls=':')
            
            plt.text(0.97,0.97,'d={:.1f}\n$f_m$={:.0f}'.format(d, fm), transform=plt.gca().transAxes, ha='right', va='top', fontsize='x-small')
    
    for i in range(nrow):
        plt.sca(ax[i,0])
        plt.ylabel('Density')
        
    for i in range(ncol):
        plt.sca(ax[ncol-1,i])
        plt.xlabel('$E_{tot}$')
    
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('../plots/grid_d_fm.pdf')
    


def batch_run():
    """"""
    
    distances = np.arange(24,28.1,1)
    distances = np.arange(26,26.1,1)
    Nradii = np.array([10,20,50,100])
    Nradii = np.array([500,])
    masses = np.array([1.8,]) * 1e10
    
    for d in distances[:]:
        for m in masses[:]:
            print(d, m)
            t1 = time.time()
            evolve_sgr(m=m*u.Msun, a=1*u.kpc, sigma_z=0*u.kpc, sigma_vz=0*u.km/u.s, d=d*u.kpc, Nr=300, logspace=True, const_arc=False, Nskip=100)
            t2 = time.time()
            print(t2-t1)

def batch_stars():
    """"""
    Nrand = 100000
    Nskip = 10
    distances = np.array([26.25, 26.5, 26.75])
    masses = np.array([1.4,])*1e10
    sizes = np.array([1,])
    comp = ['nhalo']

    for c in comp:
        for m in masses:
            for d in distances:
                for s in sizes:
                    print(c, m, d, s)
                    for i in range(Nskip):
                        t1 = time.time()
                        evolve_sgr_stars(m=m*u.Msun, d=d*u.kpc, a=s*u.kpc, Nrand=Nrand, mw_label=c, Nskip=Nskip, iskip=i)
                        t2 = time.time()
                        print(i, t2-t1)

def combine_skips():
    """"""
    # simulation setup
    d = 26.75*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    seed = 3928
    mw_label = 'nhalo'
    
    #iskip = 0
    #Nskip = 8
    #Nrand = 40000
    #mw_label = 'idisk'
    #m = 1.4e10*u.Msun
    #omega = 41*u.km/u.s/u.kpc
    #seed = 3928
    
    # number of particles
    if mw_label=='halo':
        c = initialize_halo(Nrand=Nrand, seed=seed, ret=True, graph=False)
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)
    elif mw_label=='ndisk':
        c = initialize_ndisk(Nrand=Nrand, seed=seed, ret=True, graph=False)
    elif mw_label=='nhalo':
        c = initialize_nhalo(Nrand=Nrand, seed=seed, ret=True, graph=False)
    else:
        mw_label = 'td'
        c = initialize_td(Nrand=Nrand, seed=seed, ret=True, graph=False)
    Ntot = np.size(c) + 1
    
    # read first part of HDF5 orbits
    i = 0
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}.{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip, i)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}.{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip, iskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    
    # create output HDF5 file
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    if os.path.exists(fname):
        os.remove(fname)
    fout = h5py.File(fname, 'w')
    
    # create frame group
    fout.create_group('frame')
    f['/frame'].copy(f['/frame'], fout['/frame'])
    
    # create potential and time datasets (same)
    for k in ['potential', 'time']:
        fout[k] = f[k][()]
        
    # create position and velocity datasets (to be concatenated)
    ndim, nsnap, _ = np.shape(f['pos'])
    #print(np.shape(f['pos']))
    for k in ['pos', 'vel']:
        fout.create_dataset(k, (ndim, nsnap, Ntot), dtype='<f8')
    
    f.close()
    
    c = initialize_td(ret=True, Nrand=Nrand, seed=seed)
    
    icurr = 0
    ioff = 0
    for i in range(Nskip):
        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}.{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip, i)
        fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
        #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}.{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip, i)
        #fname = '../data/bar_{:s}.h5'.format(root)
        
        f = h5py.File(fname, 'r')
        Npart = np.shape(f['pos'])[-1] - ioff
        #print(i, np.shape(f['pos']))
        #print(fname, f['pos'][0,0,1], Npart)
        
        for k in ['pos', 'vel']:
            fout[k][:,:,icurr:icurr+Npart] = f[k][:,:,ioff:]
        
        ioff = 1
        icurr += Npart
        f.close()
    
    fout.close()


# First diagnostics

def diagnose_elz():
    """"""
    
    # simulation setup
    d = 26.75*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'nhalo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #iskip = 0
    #Nskip = 8
    #Nrand = 40000
    #mw_label = 'idisk'
    #m = 2e10*u.Msun
    #omega = 41*u.km/u.s/u.kpc
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # initial
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    print(np.size(e_init), np.size(np.unique(e_init)))
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    t = Table.read('../data/rcat_giants.fits')
    if 'halo' in mw_label:
        ind_h3 = (t['eccen_pot1']>0.7) & (t['SNR']>5)
    else:
        ind_h3 = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>10)
    t = t[ind_h3]
    
    N = np.size(e_init)
    if N<10000:
        ms = 4
        alpha = 1
    else:
        ms = 1
        alpha = 0.1
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    #plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=1, mew=0, alpha=0.4)
    plt.plot(lz_init, e_init, 'ko', ms=ms, mew=0, alpha=alpha)
    
    plt.xlim(-8,2)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.plot(t['Lz']-2, t['E_tot_pot1'], 'ro', ms=1, mew=0, alpha=0.4)
    plt.plot(lz_fin, e_fin, 'ko', ms=ms, mew=0, alpha=alpha)
    plt.plot(lz_fin[0], e_fin[0], 'ro')
    
    eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    for k, e in enumerate(eridge):
        plt.axhline(e, lw=0.5, alpha=0.5)
        
    #plt.xlim(-2,2)
    #plt.ylim(-0.18,-0.08)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_{:s}.png'.format(root))

def diagnose_ehist():
    """"""
    
    # simulation setup
    d = 26.75*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'nhalo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    iskip = 0
    Nskip = 8
    Nrand = 50000
    mw_label = 'ndisk'
    m = 1e10*u.Msun
    omega = 41*u.km/u.s/u.kpc
    root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}.0'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    cgal = coord.Galactocentric(x=f['pos'][0,-1,:]*u.kpc, y=f['pos'][1,-1,:]*u.kpc, z=f['pos'][2,-1,:]*u.kpc, v_x=f['vel'][0,-1,:]*u.kpc/u.Myr, v_y=f['vel'][1,-1,:]*u.kpc/u.Myr, v_z=f['vel'][2,-1,:]*u.kpc/u.Myr)
    
    # initial
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    ind_pro = lz_fin.value<-0.5
    ind_retro = lz_fin.value>0.5
    
    # H3
    t = Table.read('../data/rcat_giants.fits')
    if 'halo' in mw_label:
        ind = (t['eccen_pot1']>0.7) & (t['SNR']>5)
        fs = 15
    else:
        ind = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
        fs = 30
    t = t[ind]
    
    #dist_err = np.median(t['dist_adpt_err']/t['dist_adpt'])
    #dist_err = 0.0
    
    ## uncertainties
    #N = np.size(cgal.x)
    #np.random.seed(281)
    
    #ceq = cgal.transform_to(coord.ICRS)
    #distance = np.abs(ceq.distance * (1 + dist_err*np.random.randn(N)))
    #ceq_err = coord.ICRS(ra=ceq.ra, dec=ceq.dec, distance=distance, radial_velocity=ceq.radial_velocity, pm_ra_cosdec=ceq.pm_ra_cosdec, pm_dec=ceq.pm_dec)
    #cgal_err = ceq_err.transform_to(coord.Galactocentric)
    
    #w0_err = gd.PhaseSpacePosition(cgal_err.transform_to(gc_frame).cartesian)
    #orbit_err = ham.integrate_orbit(w0_err, dt=0.1*u.Myr, n_steps=2)
    #etot_err = orbit_err.energy()[0]
    
    
    bins = np.linspace(-0.2,-0.02,200)
    bins = np.linspace(-0.16,-0.06,200)
    bins = np.linspace(-0.18,-0.07,100)
    
    bins_coarse = np.linspace(-0.2,-0.02,50)
    bins_fine = np.linspace(-0.2,-0.02,200)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(15,10), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(e_init.value, bins=bins, histtype='step', density=True, color='0.7', label='initial')
    plt.hist(e_fin.value, bins=bins, histtype='step', density=True, color='k', label='final')
    
    plt.legend(loc=1)
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    
    plt.hist(e_fin.value[ind_pro], bins=bins, histtype='step', density=True, color='tab:red', label='prograde')
    #if 'halo' in mw_label:
        #plt.hist(e_fin.value[ind_retro], bins=bins, histtype='step', density=True, color='tab:blue', label='retrograde')
    plt.hist(t['E_tot_pot1'], bins=bins, histtype='step', density=True, color='k', label='H3 giants')
    
    eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    for i in range(2):
        plt.sca(ax[i])
        for k, e in enumerate(eridge):
            plt.axvline(e, lw=0.5, alpha=0.5)
        
    #sigma_etot = np.median(t['E_tot_pot1_err']) * fs
    #kde = gaussian_kde(t['E_tot_pot1'], bw_method=sigma_etot)
    #x = np.linspace(-0.2,-0.02,10000)
    #y = kde(x)
    #plt.plot(x, y, 'r-', label='H3')
    
    #for i in [2,]:
        #kde = gaussian_kde(etot_err.value[::i], bw_method=sigma_etot)
        #x = np.linspace(-0.2,-0.02,10000)
        #y = kde(x)
        #plt.plot(x, y, '-', label='Model {:d}'.format(i))
    
    plt.legend(loc=1)
    plt.ylabel('Density')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/ehist_{:s}.png'.format(root))

    f.close()

def diagnose_periods():
    """"""
    
    # simulation setup
    d = 26.75*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'nhalo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #iskip = 0
    #Nskip = 8
    #Nrand = 40000
    #mw_label = 'idisk'
    #m = 2e10*u.Msun
    #omega = 41*u.km/u.s/u.kpc
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    c = coord.Galactocentric(x=orbit.pos.x[-1]*u.kpc, y=orbit.pos.y[-1]*u.kpc, z=orbit.pos.z[-1]*u.kpc, v_x=orbit.vel.d_x[-1]*u.kpc/u.Myr, v_y=orbit.vel.d_y[-1]*u.kpc/u.Myr, v_z=orbit.vel.d_z[-1]*u.kpc/u.Myr)

    if (np.size(c.x)>10001):
        skip = 5
    else:
        skip = 1
    w0 = gd.PhaseSpacePosition(c.cartesian[::skip])
    
    dt = 1*u.Myr
    Nback = 3000
    orbit_fid = ham.integrate_orbit(w0, dt=dt, n_steps=Nback)
    
    e_fin = orbit_fid.energy()[-1]
    lz_fin = orbit_fid.angular_momentum()[2][-1]
    
    # skip radial orbits that break period estimation (but keep sgr)
    #lzmin = min(lz_fin[0].value, 1)
    if d<=26*u.kpc:
        lzmin = lz_fin[0].value
    else:
        lzmin = 1
    
    ind_tangential = np.abs(lz_fin)>lzmin*u.kpc**2/u.Myr
    
    periods = orbit_fid[:,ind_tangential].estimate_period(radial=True)
    
    t = Table.read('../data/rcat_giants.fits')
    if 'halo' in mw_label:
        ind = (t['eccen_pot1']>0.7) & (t['SNR']>20)
    else:
        ind = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    
    bins = np.logspace(np.log10(75), np.log10(1020), 120)
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.hist(periods.value, bins=bins, histtype='step', density=False)
    plt.hist(t['orbit_period_pot1'][ind], bins=bins, histtype='step', density=False)
    
    pres = np.array([1, 1/2., 1/3., 1/4., 1/5., 1/6., 1/7., 1/8., 1/9., 1/10.])
    for fr in pres:
        plt.axvline(periods.value[0]*fr)
        plt.text(periods.value[0]*fr, 5.5, '{:.2f}'.format(fr))
    
    plt.gca().set_xscale('log')
    plt.xlabel('Period [Myr]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/period_{:s}.png'.format(root))
    
def aux_ratios():
    """"""
    
    a = np.array([947, 625, 467, 269, 185])
    print(a/a[0])


# Evolution following the impacts

def read_hdf5():
    """"""
    fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.010000.h5'
    fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'
    f = h5py.File(fname, 'r')
    all_snaps = list(f.keys())
    #print(len(all_snaps))
    
    snaps = all_snaps[::10]
    Nsnap = len(snaps)
    da = 3
    
    plt.close()
    fig, ax = plt.subplots(1, Nsnap, figsize=(Nsnap*da*0.8, da), sharex=True, sharey=True)
    
    for i in range(Nsnap):
        plt.sca(ax[i])
        c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        #w0_ = gd.PhaseSpacePosition(c.cartesian)
        #orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        #etot = orbit_.energy()[0,:]
        #lz = orbit_.angular_momentum()[2,0,:]
        
        #plt.plot(lz, etot, 'o', mew=0, ms=1, alpha=0.1)
        
        plt.plot(c.x, c.y, 'o', mew=0, ms=1, alpha=0.1)
    
        plt.xlim(-30,30)
        plt.ylim(-30,30)
        plt.gca().set_aspect('equal')
        plt.xlabel('x [kpc]')
    
    plt.sca(ax[0])
    plt.ylabel('y [kpc]')
    plt.tight_layout(h_pad=0, w_pad=0)

    f.close()

def elz_evolution(verbose=False):
    """Plot ELz in individual snapshots"""
    
    fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.010000.h5'
    #fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'
    fname = '../data/snaps_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.h5'
    fname = '../data/snaps_interact0_d.26.6_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact1_lmc_d.27.0_m.25.00.0.00_a.5.0.1.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact0_lmc_d.27.0_m.25.00.1.00_a.5.0.1.0_ndisk_N.100000.h5'
    f = h5py.File(fname, 'r')
    snaps = list(f.keys())
    
    Nsnap = len(snaps)
    Nrow = int(Nsnap // np.sqrt(Nsnap))
    Ncol = int(np.ceil(Nsnap/Nrow))
    
    if verbose: print(Nsnap, Nrow, Ncol)
    da = 3.5
    
    T = -6490.5*u.Myr
    T = -2977.5*u.Myr
    T = -1111.5*u.Myr
    T = -3413.5*u.Myr
    dt = 0.5*u.Myr
    nskip = 200
    
    #pkl = pickle.load(open('../data/sgr_orbit_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.010000.pkl', 'rb'))
    pkl = pickle.load(open('../data/sgr_orbit_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.pkl', 'rb'))
    pkl = pickle.load(open('../data/sgr_orbit_interact0_d.26.6_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.pkl', 'rb'))
    #pkl = pickle.load(open('../data/sgr_orbit_interact1_lmc_d.27.0_m.25.00.0.00_a.5.0.1.0_ndisk_N.100000.pkl', 'rb'))
    #pkl = pickle.load(open('../data/sgr_orbit_interact0_lmc_d.27.0_m.25.00.1.00_a.5.0.1.0_ndisk_N.100000.pkl', 'rb'))
    
    c = pkl['c'][::-1]
    t = pkl['t'][::-1]
    q = gd.PhaseSpacePosition(c.cartesian)
    
    epot = ham.potential.energy(q)
    #print(epot)
    #print(t)
    
    epot_plot = epot[::nskip]
    
    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(Ncol*da, Nrow*da), sharex=True, sharey=True)
    
    for i in range(Nsnap):
        irow = i//Ncol
        icol = i%Ncol
        plt.sca(ax[irow, icol])
        c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        w0_ = gd.PhaseSpacePosition(c.cartesian)
        orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        etot = orbit_.energy()[0,:]
        lz = orbit_.angular_momentum()[2,0,:]
        
        plt.plot(lz, etot, 'ko', mew=0, ms=0.5, alpha=0.1)
        plt.axhline(epot_plot[i].value, color='r', alpha=0.4)
        
        t = T + dt*nskip*i
        plt.text(0.9,0.9,'{:.2f} Gyr'.format(t.to(u.Gyr).value), transform=plt.gca().transAxes, ha='right', va='top', fontsize='small')
        
        if irow==Nrow-1:
            plt.xlabel('$L_z$ kpc$^2$ Myr$^{-1}$')
        
        if icol==0:
            plt.ylabel('$E_{tot}$ kpc$^2$ Myr$^{-2}$')
    
    plt.xlim(-6,1.5)
    plt.ylim(-0.185,-0.06)
    plt.tight_layout(h_pad=0, w_pad=0)
    
    #plt.savefig('../plots/elz_evolution.png')

def pot(R):
    return ham.potential.energy([R, 0, 0]*u.kpc).value[0]

pot_vec = np.vectorize(pot)

def Lcirc(Etot, R):
    return -R*((2*(Etot - pot_vec(R)))**0.5) 

def maxLcirc(Etot):
    optfunc = partial(Lcirc,Etot)
    res = minimize(optfunc, np.array([0.1]), method='BFGS')
    return np.abs(res.fun)


def ehist_evolution(verbose=False):
    """Plot energy histogram in individual snapshots"""
    
    fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.010000.h5'
    #fname = '../data/snaps_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'
    fname = '../data/snaps_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.h5'
    fname = '../data/snaps_interact1_lmc_d.27.0_m.25.00.0.00_a.5.0.1.0_ndisk_N.100000.h5'
    fname = '../data/snaps_interact1_lmc_d.27.0_m.8.00.0.00_a.5.0.1.0_ndisk_N.100000.h5'
    f = h5py.File(fname, 'r')
    snaps = list(f.keys())
    
    Nsnap = len(snaps)
    Nrow = int(Nsnap // np.sqrt(Nsnap))
    Ncol = int(np.ceil(Nsnap/Nrow))
    
    if verbose: print(Nsnap, Nrow, Ncol)
    da = 3.5
    
    T = -6490.5*u.Myr
    T = -2977.5*u.Myr
    dt = 0.5*u.Myr
    nskip = 200
    
    #pkl = pickle.load(open('../data/sgr_orbit_interact0_d.27.0_m.1.40.16.00_a.1.0.5.0_ndisk_N.010000.pkl', 'rb'))
    pkl = pickle.load(open('../data/sgr_orbit_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.pkl', 'rb'))
    pkl = pickle.load(open('../data/sgr_orbit_interact1_lmc_d.27.0_m.8.00.0.00_a.5.0.1.0_ndisk_N.100000.pkl', 'rb'))
    
    c = pkl['c'][::-1]
    t = pkl['t'][::-1]
    q = gd.PhaseSpacePosition(c.cartesian)
    
    epot = ham.potential.energy(q)
    #print(epot)
    #print(t)
    
    epot_plot = epot[::nskip]
    ebins = np.linspace(-0.18,-0.08,100)
    eridge = np.array([-0.145, -0.134, -0.127, -0.122, -0.116])
    
    # read in H3 data
    th = Table.read('../data/rcat_giants.fits')
    ind = (th['SNR']>3)
    th = th[ind]
    
    ind_disk = (th['Lz']<0) & (th['circLz_pot1']>0.35) & (th['circLz_pot1']<0.7)
    #ind_disk = (th['Lz']<0) & (th['circLz_pot1']<0.7)

    plt.close()
    fig, ax = plt.subplots(Nrow, Ncol, figsize=(Ncol*da, Nrow*da), sharex=True, sharey=True)
    
    for i in range(Nsnap):
        irow = i//Ncol
        icol = i%Ncol
        plt.sca(ax[irow, icol])
        c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        w0_ = gd.PhaseSpacePosition(c.cartesian)
        orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        etot = orbit_.energy()[0,:]
        lz = orbit_.angular_momentum()[2,0,:]
        
        ## calculate circularity
        #maxLcirc_vec = np.vectorize(maxLcirc)
        #maxLcirc_arr = maxLcirc_vec(np.linspace(-0.2, -0.02, 1000))
        
        #lmax = (np.interp(etot.value, np.linspace(-0.2, 0.02, 1000), maxLcirc_arr))
        #circLz = np.abs(lz.value/lmax)
        
        #ind = circLz>0.35
        
        plt.hist(etot.value, bins=ebins, color='k', alpha=0.8, histtype='step', density=True)
        #plt.hist(th['E_tot_pot2'][ind_disk], color='orange', bins=ebins, density=True, histtype='stepfilled', alpha=0.3, label='H3 disk (0.3<circ<0.7)')
        
        plt.axvline(epot_plot[i].value, color='r', alpha=0.4)
        
        for e in eridge:
            plt.axvline(e, ls=':', color='k', alpha=0.5, lw=0.5)
        
        t = T + dt*nskip*i
        plt.text(0.9,0.9,'{:.2f} Gyr'.format(t.to(u.Gyr).value), transform=plt.gca().transAxes, ha='right', va='top', fontsize='small')
        
        if irow==Nrow-1:
            plt.xlabel('$E_{tot}$ kpc$^2$ Myr$^{-2}$')
        
        if icol==0:
            plt.ylabel('Density')
    
    #plt.xlim(-6,1.5)
    #plt.xlim(-0.185,-0.06)
    plt.xlim(-0.15,-0.095)
    plt.tight_layout(h_pad=0, w_pad=0)
    
    #plt.savefig('../plots/ehist_evolution.png')


def visualize_lz_change():
    """"""
    
    # simulation setup
    d = 26.5*u.kpc
    m = 5e10*u.Msun
    a = 2*u.kpc
    Nskip = 8
    Nrand = 40000
    mw_label = 'idisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #iskip = 0
    #Nskip = 8
    #Nrand = 40000
    #mw_label = 'idisk'
    #m = 2e10*u.Msun
    #omega = 41*u.km/u.s/u.kpc
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # initial
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    de = e_fin/e_init
    
    dlz = np.abs(lz_fin/lz_init)
    dlz[dlz<0.05] = 0.05
    dlz[(dlz>0.95) | (~np.isfinite(dlz))] = 0.95
    #dlz[dlz>0.8] = 1
    
    tout = Table([orbit.pos.x[-1], orbit.pos.y[-1], orbit.pos.z[-1], e_fin, lz_fin, de, dlz], names=('x', 'y', 'z', 'e', 'lz', 'de', 'dlz'))
    tout.write('../data/sgr_now_{:s}.csv'.format(root), overwrite=True)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    plt.scatter(orbit.pos.x[0], orbit.pos.y[0], c=e_fin.value, ec='none', s=5, cmap='jet', vmin=-0.2, vmax=-0.05)
    #plt.scatter(orbit.pos.x[0], orbit.pos.y[0], c=e_init.value, ec='none', s=5, cmap='jet', vmin=-0.2, vmax=-0.05)
    
    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    
    plt.sca(ax[1])
    plt.scatter(orbit.pos.x[-1], orbit.pos.y[-1], c=e_fin.value, ec='none', alpha=1-dlz, s=5, cmap='jet', vmin=-0.2, vmax=-0.05)
    
    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    
    #plt.sca(ax[2])
    #t = Table.read('../data/rcat_giants.fits')
    ##ind = t['SNR']>snr
    ##t = t[ind]
    
    #ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) #& (t['SNR']>20)
    #ind_halo = (t['eccen_pot1']>0.7) & (t['Lz']<0.) #& (t['SNR']>20) # & (t['FeH']<-1)
    #t = t[ind_disk]
    
    #plt.plot(t['X_gal'], t['Y_gal'], 'k.', ms=1)
    #plt.xlim(-30,30)
    #plt.ylim(-30,30)
    #plt.gca().set_aspect('equal', adjustable='datalim')
    #plt.xlabel('x [kpc]')
    #plt.ylabel('y [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/xy_dlz_{:s}.png'.format(root))

def paraview_lz_change():
    """"""
    
    # simulation setup
    d = 26.5*u.kpc
    m = 5e10*u.Msun
    a = 2*u.kpc
    Nskip = 8
    Nrand = 40000
    mw_label = 'idisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # initial
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    de = e_fin/e_init
    
    dlz = np.abs(lz_fin/lz_init)
    dlz[dlz<0.05] = 0.05
    dlz[(dlz>0.95) | (~np.isfinite(dlz))] = 0.95
    
    Nt = np.shape(orbit.pos.x)[0]
    for i in range(Nt):
        tout = Table([0.1*orbit.pos.x[i], 0.1*orbit.pos.y[i], 0.1*orbit.pos.z[i], e_fin, lz_fin, de, dlz], names=('x', 'y', 'z', 'e', 'lz', 'de', 'dlz'))
        tout.write('../data/paraview/sgr_{:s}_{:03d}.csv'.format(root, i), overwrite=True)

def boot_bar_ehist():
    """"""
    
    Nskip = 20
    Nrand = 50000
    mw_label = 'halo'
    m = 0.5e10*u.Msun
    omega = 41*u.km/u.s/u.kpc
    root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    cgal = coord.Galactocentric(x=f['pos'][0,-1,:]*u.kpc, y=f['pos'][1,-1,:]*u.kpc, z=f['pos'][2,-1,:]*u.kpc, v_x=f['vel'][0,-1,:]*u.kpc/u.Myr, v_y=f['vel'][1,-1,:]*u.kpc/u.Myr, v_z=f['vel'][2,-1,:]*u.kpc/u.Myr)
    
    # initial
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    snr = 20
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>snr
    t = t[ind]
    N = len(t)
    
    if mw_label=='td':
        ind = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    else:
        ind = (t['eccen_pot1']>0.7) & (t['Lz']<0.)
    
    # randomly draw 6D positions
    Nboot = 1000
    Ndim = 6
    seed = 1248
    np.random.seed(seed)
    offsets = np.random.randn(N,Nboot,Ndim)
    
    ra = (t['GAIAEDR3_RA'][:,np.newaxis] + offsets[:,:,0] * t['GAIAEDR3_RA_ERROR'][:,np.newaxis]) * u.deg
    dec = (t['GAIAEDR3_DEC'][:,np.newaxis] + offsets[:,:,1] * t['GAIAEDR3_DEC_ERROR'][:,np.newaxis]) * u.deg
    dist = (t['dist_adpt'][:,np.newaxis] + offsets[:,:,2] * t['dist_adpt_err'][:,np.newaxis]) * u.kpc
    dist[dist<0*u.kpc] = 0*u.kpc
    pmra = (t['GAIAEDR3_PMRA'][:,np.newaxis] + offsets[:,:,3] * t['GAIAEDR3_PMRA_ERROR'][:,np.newaxis]) * u.mas/u.yr
    pmdec = (t['GAIAEDR3_PMDEC'][:,np.newaxis] + offsets[:,:,4] * t['GAIAEDR3_PMDEC_ERROR'][:,np.newaxis]) * u.mas/u.yr
    vr = (t['Vrad'][:,np.newaxis] + offsets[:,:,5] * t['Vrad_err'][:,np.newaxis]) * u.km/u.s
    
    # calculate orbital properties
    c = coord.SkyCoord(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=0)
    
    etot = orbit.energy()[0].reshape(N,-1)
    lz = orbit.angular_momentum()[2][0].reshape(N,-1)
    
    ebins = np.linspace(-0.18,-0.05,150)
    
    # plotting
    color_disk = mpl.cm.Reds(0.5)
    #color_halo = mpl.cm.Blues((i+1)/Nsnr)
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.hist(etot[ind].value.flatten(), bins=ebins, density=True, histtype='step', color=color_disk, label='H3 S/N={:d}'.format(snr))
    plt.hist(e_fin.value, bins=ebins, histtype='step', density=True, color='k', label='Bar model')
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'Halo\n(ecc>0.7) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.legend(fontsize='small', bbox_to_anchor=(0.99,0.8), loc=1)
    
    plt.tight_layout()


def elz_time():
    """Plot E, Lz vs time"""
    
    # simulation setup
    d = 28*u.kpc
    m = 2.8e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'ndisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # select a narrow bin of final energies
    e_fin = orbit.energy()[-1]
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    elevels = np.array([-0.1282, -0.1225])*u.kpc**2*u.Myr**-2
    elevels = np.array([-0.126, -0.124])*u.kpc**2*u.Myr**-2
    #elevels = np.array([-0.0883, -0.0796])*u.kpc**2*u.Myr**-2
    ind_e = (e_fin>elevels[0]) & (e_fin<elevels[1])
    
    etot = orbit.energy()
    lz = orbit.angular_momentum()[2]
    
    print(np.shape(etot[:,ind_e]))
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(orbit.t, etot[:,ind_e], 'k-', lw=0.5, alpha=0.1)
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    
    plt.sca(ax[1])
    plt.plot(orbit.t, lz[:,ind_e], 'k-', lw=0.5, alpha=0.1)
    plt.ylabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('Time [Myr]')

    plt.tight_layout()
    plt.savefig('../plots/e_lz_time_{:s}.png'.format(root))

def elz_time_hist():
    """Show the initial and final distributions of E,Lz"""
    
    # simulation setup
    d = 27*u.kpc
    m = 2.8e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'ndisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # calculate orbital properties
    etot = orbit.energy()
    lz = orbit.angular_momentum()[2]
    
    # select a narrow bin of final energies
    elevels = np.array([-0.1391, -0.133, -0.1286, -0.1231, -0.1123, -0.1026,])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    ind_energy = [(etot[-1]>elevels[2*x]) & (etot[-1]<elevels[2*x+1]) for x in range(Nlevel)]
    
    # bins
    e_bins = np.linspace(-0.15,-0.09,50)
    lz_bins = np.linspace(-4,0,50)
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(12,8))
    
    for i in range(Nlevel):
        plt.sca(ax[0,i])
        plt.hist(etot.value[0,ind_energy[i]], bins=e_bins, histtype='step', label='Initial')
        plt.hist(etot.value[-1,ind_energy[i]], bins=e_bins, histtype='step', label='Final')
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.ylabel('Number')
    
        plt.sca(ax[1,i])
        plt.hist(lz.value[0,ind_energy[i]], bins=lz_bins, histtype='step', label='Initial')
        plt.hist(lz.value[-1,ind_energy[i]], bins=lz_bins, histtype='step', label='Final')
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('Number')
    
    plt.sca(ax[0,0])
    plt.legend(fontsize='x-small', loc=1)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_hist_evo_{:s}.png'.format(root))

def elz_tip():
    """Explore the original locations of stars at the tip of the ripple"""
    
    # simulation setup
    d = 27*u.kpc
    m = 2.8e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'ndisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # calculate orbital properties
    etot = orbit.energy()
    lz = orbit.angular_momentum()[2]
    
    # select a narrow bin of final energies
    elevels = np.array([-0.1391, -0.133, -0.1286, -0.1231, -0.1123, -0.1026,])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    ind_energy = [(etot[-1]>elevels[2*x]) & (etot[-1]<elevels[2*x+1]) for x in range(Nlevel)]
    
    print(np.shape(lz))
    print(np.shape(lz.value[0, ind_energy[0]]))
    #print(lz)
    
    ind_lz = lz[-1]>-1.1#*u.kpc**2*u.Myr**-1
    lz_bins = np.linspace(-4,0,50)

    plt.close()
    plt.figure()
    
    i = 1
    #plt.plot(lz.value[-1,:], etot.value[-1,:], 'ko', ms=1)
    #plt.plot(lz.value[-1, ind_energy[i]], etot.value[-1, ind_energy[i]], 'ro', ms=1)
    
    #plt.hist(lz.value[-1, ind_energy[i] & ind_lz], histtype='step', density=True)
    plt.hist(lz.value[0, ind_energy[i]], bins=lz_bins, histtype='step', density=True)
    plt.hist(lz.value[0, ind_energy[i] & ind_lz], bins=lz_bins, histtype='step', density=True)
    
    plt.tight_layout()


# Spatial dependence

def elz_sides():
    """Plot resulting ELz in different halves of the galaxy"""
    
    # simulation setup
    d = 27*u.kpc
    m = 2.8e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'ndisk'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    #iskip = 0
    #Nskip = 8
    #Nrand = 40000
    #mw_label = 'idisk'
    #m = 2e10*u.Msun
    #omega = 41*u.km/u.s/u.kpc
    #root = 'm.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip)
    #fname = '../data/bar_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    # initial
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    print(np.size(e_init), np.size(np.unique(e_init)))
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    t = Table.read('../data/rcat_giants.fits')
    if 'halo' in mw_label:
        ind_h3 = (t['eccen_pot1']>0.7) & (t['SNR']>5)
    else:
        ind_h3 = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>10)
    t = t[ind_h3]
    
    N = np.size(e_init)
    if N<10000:
        ms = 4
        alpha = 1
    else:
        ms = 1
        alpha = 0.1
    
    ind_xp = orbit.pos.x[-1]>0
    ind_yp = orbit.pos.y[-1]>0
    ind_zp = orbit.pos.z[-1]>0
    
    selections = [ind_xp, ind_yp, ind_zp]
    labels_positive = ['x>0', 'y>0', 'z>0']
    labels_negative = ['x<0', 'y<0', 'z<0']
    
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(12,8), sharex=True, sharey=True)
    
    for i in range(3):
        plt.sca(ax[0,i])
        plt.plot(lz_fin[selections[i]], e_fin[selections[i]], 'ko', ms=ms, mew=0, alpha=alpha)
        plt.text(0.9,0.9,labels_positive[i], transform=plt.gca().transAxes, ha='right', va='top', fontsize='small')
        
        plt.sca(ax[1,i])
        plt.plot(lz_fin[~selections[i]], e_fin[~selections[i]], 'ko', ms=ms, mew=0, alpha=alpha)
        plt.text(0.9,0.9,labels_negative[i], transform=plt.gca().transAxes, ha='right', va='top', fontsize='small')
        
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.xlim(-8,2)
    plt.ylim(-0.18, -0.03)
    
    for i in range(2):
        plt.sca(ax[i,0])
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    for ax_ in ax.ravel():
        plt.sca(ax_)
        for k, e in enumerate(eridge):
            plt.axhline(e, lw=0.5, alpha=0.5)
        
    ##plt.xlim(-2,2)
    ##plt.ylim(-0.18,-0.08)
    #plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_xyz_{:s}.png'.format(root))

def localization():
    """Check if ripples are localized at early times"""
    
    fname = '../data/snaps_interact0_lmc_d.27.0_m.25.00.1.00_a.5.0.1.0_ndisk_N.100000.h5'
    f = h5py.File(fname, 'r')
    snaps = list(f.keys())
    Nsnap = len(snaps)
    
    ncol = 5
    psnaps = np.linspace(0, Nsnap-1, ncol, dtype=int)
    #psnaps = np.arange(ncol, dtype=int)
    #print(psnaps)
    
    i = psnaps[1]
    c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
    w0_ = gd.PhaseSpacePosition(c.cartesian)
    orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
    etot = orbit_.energy()[0,:]
    lz = orbit_.angular_momentum()[2,0,:]
    
    psnaps = [0,2,4,6,9]
    
    ind = (etot>-0.144*u.kpc**2*u.Myr**-2) & (etot<-0.14*u.kpc**2*u.Myr**-2) & (lz>-1.4*u.kpc**2*u.Myr**-1)
    ind2 = (etot>-0.135*u.kpc**2*u.Myr**-2) & (etot<-0.13*u.kpc**2*u.Myr**-2) & (lz>-1.6*u.kpc**2*u.Myr**-1)
    ind3 = (etot>-0.127*u.kpc**2*u.Myr**-2) & (etot<-0.121*u.kpc**2*u.Myr**-2) & (lz>-1.9*u.kpc**2*u.Myr**-1)
    
    da = 2
    fwidth = ncol*da
    fheight = 2*da
    
    ms = 1
    
    plt.close()
    fig, ax = plt.subplots(2,ncol,figsize=(fwidth, fheight), sharex='row', sharey='row')
    
    for i in range(ncol):
        j = psnaps[i]
        c = coord.Galactocentric(x=(f[snaps[j]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[j]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[j]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[j]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[j]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[j]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        w0_ = gd.PhaseSpacePosition(c.cartesian)
        orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        etot = orbit_.energy()[0,:]
        lz = orbit_.angular_momentum()[2,0,:]
        
        #ind = (etot>-0.146*u.kpc**2*u.Myr**-2) & (etot<-0.141*u.kpc**2*u.Myr**-2) & (lz>-1.3*u.kpc**2*u.Myr**-1)
        
        plt.sca(ax[0][i])
        plt.plot(c.x, c.y, 'ko', mew=0, ms=0.5, alpha=0.1)
        #if i==ncol-1:
        plt.plot(c.x[ind], c.y[ind], 'ro', mew=0, ms=ms, alpha=1)
        plt.plot(c.x[ind2], c.y[ind2], 'bo', mew=0, ms=ms, alpha=1)
        plt.plot(c.x[ind3], c.y[ind3], 'go', mew=0, ms=ms, alpha=1)
        
        plt.sca(ax[1][i])
        plt.plot(lz, etot, 'ko', mew=0, ms=0.5, alpha=0.1)
        #if i==ncol-1:
        plt.plot(lz[ind], etot[ind], 'ro', mew=0, ms=ms, alpha=1)
        plt.plot(lz[ind2], etot[ind2], 'bo', mew=0, ms=ms, alpha=1)
        plt.plot(lz[ind3], etot[ind3], 'go', mew=0, ms=ms, alpha=1)
    
    plt.sca(ax[0][0])
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    
    plt.sca(ax[1][0])
    plt.xlim(-6,1.5)
    plt.ylim(-0.185,-0.06)
    
    plt.tight_layout(h_pad=0, w_pad=0)

def init_r():
    """Plot where stars end up depending on their initial distance from the galactic center"""
    
    fname = '../data/snaps_interact0_lmc_d.27.0_m.25.00.1.00_a.5.0.1.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact0_d.26.6_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'

    f = h5py.File(fname, 'r')
    snaps = list(f.keys())
    Nsnap = len(snaps)
    
    ncol = 5
    psnaps = np.linspace(0, Nsnap-1, ncol, dtype=int)
    #psnaps = np.arange(ncol, dtype=int)
    #print(psnaps)
    
    i = 0
    c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
    
    r = c.spherical.distance.value
    
    #w0_ = gd.PhaseSpacePosition(c.cartesian)
    #orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
    #etot = orbit_.energy()[0,:]
    #lz = orbit_.angular_momentum()[2,0,:]
    
    #psnaps = [0,2,4,6,9]
    
    #ind = (etot>-0.144*u.kpc**2*u.Myr**-2) & (etot<-0.14*u.kpc**2*u.Myr**-2) & (lz>-1.4*u.kpc**2*u.Myr**-1)
    #ind2 = (etot>-0.135*u.kpc**2*u.Myr**-2) & (etot<-0.13*u.kpc**2*u.Myr**-2) & (lz>-1.6*u.kpc**2*u.Myr**-1)
    #ind3 = (etot>-0.127*u.kpc**2*u.Myr**-2) & (etot<-0.121*u.kpc**2*u.Myr**-2) & (lz>-1.9*u.kpc**2*u.Myr**-1)
    
    da = 2.5
    fwidth = ncol*da
    fheight = 2.3*da
    
    s = 0.5
    alpha = 0.1
    vmin = 0
    vmax = 20
    
    plt.close()
    fig, ax = plt.subplots(2,ncol,figsize=(fwidth, fheight), sharex='row', sharey='row')
    
    for i in range(ncol):
        j = psnaps[i]
        c = coord.Galactocentric(x=(f[snaps[j]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[j]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[j]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[j]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[j]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[j]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        w0_ = gd.PhaseSpacePosition(c.cartesian)
        orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        etot = orbit_.energy()[0,:]
        lz = orbit_.angular_momentum()[2,0,:]
        
        r = c.spherical.distance.value
        
        #ind = (etot>-0.146*u.kpc**2*u.Myr**-2) & (etot<-0.141*u.kpc**2*u.Myr**-2) & (lz>-1.3*u.kpc**2*u.Myr**-1)
        
        plt.sca(ax[0][i])
        plt.scatter(c.x, c.y, c=r, s=s, vmin=vmin, vmax=vmax, alpha=alpha)
        ##if i==ncol-1:
        #plt.plot(c.x[ind], c.y[ind], 'ro', mew=0, ms=ms, alpha=1)
        #plt.plot(c.x[ind2], c.y[ind2], 'bo', mew=0, ms=ms, alpha=1)
        #plt.plot(c.x[ind3], c.y[ind3], 'go', mew=0, ms=ms, alpha=1)
        
        plt.xlabel('X [kpc]')
        
        plt.sca(ax[1][i])
        plt.scatter(lz, etot, c=r, s=s, vmin=vmin, vmax=vmax, alpha=alpha)
        ##if i==ncol-1:
        #plt.plot(lz[ind], etot[ind], 'ro', mew=0, ms=ms, alpha=1)
        #plt.plot(lz[ind2], etot[ind2], 'bo', mew=0, ms=ms, alpha=1)
        #plt.plot(lz[ind3], etot[ind3], 'go', mew=0, ms=ms, alpha=1)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.sca(ax[0][0])
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.ylabel('Y [kpc]')
    
    plt.sca(ax[1][0])
    plt.xlim(-6,1.5)
    plt.ylim(-0.185,-0.06)
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout(h_pad=0, w_pad=0)

def init_phi():
    """Plot where stars end up depending on their initial distance from the galactic center"""
    
    fname = '../data/snaps_interact0_lmc_d.27.0_m.25.00.1.00_a.5.0.1.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact1_d.26.6_m.0.05.0.21_a.0.5.10.0_ndisk_N.100000.h5'
    #fname = '../data/snaps_interact0_d.26.6_m.1.40.16.00_a.1.0.5.0_ndisk_N.100000.h5'

    f = h5py.File(fname, 'r')
    snaps = list(f.keys())
    Nsnap = len(snaps)
    
    ncol = 5
    psnaps = np.linspace(0, Nsnap-1, ncol, dtype=int)
    #psnaps = np.arange(ncol, dtype=int)
    #print(psnaps)
    
    i = 0
    c = coord.Galactocentric(x=(f[snaps[i]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[i]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[i]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[i]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[i]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[i]]['vz'][:]*u.m/u.s).to(u.km/u.s))
    
    phi = c.spherical.lon.deg
    
    #w0_ = gd.PhaseSpacePosition(c.cartesian)
    #orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
    #etot = orbit_.energy()[0,:]
    #lz = orbit_.angular_momentum()[2,0,:]
    
    #psnaps = [0,2,4,6,9]
    
    #ind = (etot>-0.144*u.kpc**2*u.Myr**-2) & (etot<-0.14*u.kpc**2*u.Myr**-2) & (lz>-1.4*u.kpc**2*u.Myr**-1)
    #ind2 = (etot>-0.135*u.kpc**2*u.Myr**-2) & (etot<-0.13*u.kpc**2*u.Myr**-2) & (lz>-1.6*u.kpc**2*u.Myr**-1)
    #ind3 = (etot>-0.127*u.kpc**2*u.Myr**-2) & (etot<-0.121*u.kpc**2*u.Myr**-2) & (lz>-1.9*u.kpc**2*u.Myr**-1)
    
    da = 2.5
    fwidth = ncol*da
    fheight = 2.3*da
    
    s = 0.2
    alpha = 0.1
    vmin = 0
    vmax = 360
    
    plt.close()
    fig, ax = plt.subplots(2,ncol,figsize=(fwidth, fheight), sharex='row', sharey='row')
    
    for i in range(ncol):
        j = psnaps[i]
        c = coord.Galactocentric(x=(f[snaps[j]]['x'][:]*u.m).to(u.kpc), y=(f[snaps[j]]['y'][:]*u.m).to(u.kpc), z=(f[snaps[j]]['z'][:]*u.m).to(u.kpc), v_x=(f[snaps[j]]['vx'][:]*u.m/u.s).to(u.km/u.s), v_y=(f[snaps[j]]['vy'][:]*u.m/u.s).to(u.km/u.s), v_z=(f[snaps[j]]['vz'][:]*u.m/u.s).to(u.km/u.s))
        
        w0_ = gd.PhaseSpacePosition(c.cartesian)
        orbit_ = ham.integrate_orbit(w0_, dt=0.1*u.Myr, n_steps=2)
        etot = orbit_.energy()[0,:]
        lz = orbit_.angular_momentum()[2,0,:]
        
        #ind = (etot>-0.146*u.kpc**2*u.Myr**-2) & (etot<-0.141*u.kpc**2*u.Myr**-2) & (lz>-1.3*u.kpc**2*u.Myr**-1)
        
        plt.sca(ax[0][i])
        plt.scatter(c.x, c.y, c=phi, s=s, vmin=vmin, vmax=vmax, alpha=alpha)
        ##if i==ncol-1:
        #plt.plot(c.x[ind], c.y[ind], 'ro', mew=0, ms=ms, alpha=1)
        #plt.plot(c.x[ind2], c.y[ind2], 'bo', mew=0, ms=ms, alpha=1)
        #plt.plot(c.x[ind3], c.y[ind3], 'go', mew=0, ms=ms, alpha=1)
        
        plt.xlabel('X [kpc]')
        
        plt.sca(ax[1][i])
        plt.scatter(lz, etot, c=phi, s=s, vmin=vmin, vmax=vmax, alpha=alpha)
        ##if i==ncol-1:
        #plt.plot(lz[ind], etot[ind], 'ro', mew=0, ms=ms, alpha=1)
        #plt.plot(lz[ind2], etot[ind2], 'bo', mew=0, ms=ms, alpha=1)
        #plt.plot(lz[ind3], etot[ind3], 'go', mew=0, ms=ms, alpha=1)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.sca(ax[0][0])
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.ylabel('Y [kpc]')
    
    plt.sca(ax[1][0])
    plt.xlim(-6,1.5)
    plt.ylim(-0.185,-0.06)
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout(h_pad=0, w_pad=0)



# Resulting distributions

def ehist_distance_halo():
    """"""
    
    # H3
    t = Table.read('../data/rcat_giants.fits')
    ind_gse = (t['eccen_pot1']>0.7) & (t['SNR']>3) # & (t['FeH']<-1)
    t = t[ind_gse]
    
    # simulation setup
    m = 1.4e10*u.Msun
    a = 7*u.kpc
    Nskip = 10
    Nrand = 50000
    mw_label = 'td'
    
    bins = np.linspace(-0.2,-0.02,200)
    bins_wide = np.linspace(-0.2,-0.02,120)
    distances = np.arange(24,29,1)
    distances = np.array([21,24,27])
    Nd = np.size(distances)

    plt.close()
    fig, ax = plt.subplots(Nd, 1, figsize=(12,8), sharex=True)

    for e, dist in enumerate(distances):
        d = dist*u.kpc
        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
        fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
        
        f = h5py.File(fname, 'r')
        orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
        f.close()
        
        e_init = orbit.energy()[0]
        e_fin = orbit.energy()[-1]
        epot = np.min(orbit.potential_energy(), axis=0)
        #print(np.shape(e_fin), np.shape(epot))
        
        plt.sca(ax[e])
        plt.hist(e_fin[1:].value, bins=bins, histtype='step', density=True, color=mpl.cm.Blues_r(e/6), alpha=0.3)
        #plt.hist(e_init[1:].value, bins=bins, histtype='step', density=True, color=mpl.cm.Blues_r(e/6), alpha=0.3)
        plt.hist(t['E_tot_pot1'], bins=bins_wide, histtype='step', density=True, color='r', alpha=0.3)
        
        esgr = e_fin[0].value
        plt.axvline(esgr, color='k', lw=0.5)
        #plt.axvline(epot[0].value, color='0.5')
        
        #for f in [1.9,1.6,1.3]:
            #plt.axvline(f*esgr, color='0.5', lw=0.5)
        
        
        plt.ylabel('Density')
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.tight_layout(h_pad=0)

def ehist_disk_halo():
    """"""
    
    # simulation setup
    m = 1.4e10*u.Msun
    a = 7*u.kpc
    Nskip = 10
    Nrand = 50000
    
    bins = np.linspace(-0.2,-0.02,200)
    bins_wide = np.linspace(-0.2,-0.02,120)
    distances = np.array([21,24,27])
    Nd = np.size(distances)
    
    plt.close()
    fig, ax = plt.subplots(Nd, 1, figsize=(12,8), sharex=True)

    for e, dist in enumerate(distances):
        for mw_label in ['halo', 'td']:
            d = dist*u.kpc
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
            
            f = h5py.File(fname, 'r')
            orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
            f.close()
            
            e_init = orbit.energy()[0]
            e_fin = orbit.energy()[-1]
            epot = np.min(orbit.potential_energy(), axis=0)
            #print(np.shape(e_fin), np.shape(epot))
            
            plt.sca(ax[e])
            plt.hist(e_fin[1:].value, bins=bins, histtype='step', density=True, alpha=0.3)
            #plt.hist(e_init[1:].value, bins=bins, histtype='step', density=True, color=mpl.cm.Blues_r(e/6), alpha=0.3)
            #plt.hist(t['E_tot_pot1'], bins=bins_wide, histtype='step', density=True, color='r', alpha=0.3)
        
        esgr = e_fin[0].value
        plt.axvline(esgr, color='k', lw=0.5)
        
        plt.ylabel('Density')
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.tight_layout(h_pad=0)

def diagnose_rvr(d=27*u.kpc):
    """"""
    
    # simulation setup
    #d = 26.5*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 100000
    mw_label = 'nhalo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    c = coord.Galactocentric(x=orbit.pos.x[-1]*u.kpc, y=orbit.pos.y[-1]*u.kpc, z=orbit.pos.z[-1]*u.kpc, v_x=orbit.vel.d_x[-1]*u.kpc/u.Myr, v_y=orbit.vel.d_y[-1]*u.kpc/u.Myr, v_z=orbit.vel.d_z[-1]*u.kpc/u.Myr)
    
    r = orbit.spherical.distance*u.kpc
    vr = (orbit.spherical.vel.d_distance*u.kpc/u.Myr).to(u.km/u.s)
    
    print(np.shape(r), np.shape(vr))
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)
    
    for e, i in enumerate([0,-1]):
        plt.sca(ax[e])
        plt.plot(r[i], vr[i], 'ko', ms=1, mew=0, alpha=0.5)
        
        plt.xlabel('r [kpc]')
    
    plt.sca(ax[0])
    plt.xlim(1,100)
    plt.ylim(-400,400)
    plt.ylabel('$V_r$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/rvr_{:s}.png'.format(root))



def suite_kde():
    """"""
    
    # simulation setup
    Nskip = 10
    Nrand = 50000
    a = 7*u.kpc
    d0 = 24*u.kpc
    m0 = 1.4e10*u.Msun
    masses = np.array([0.7,1.4,2.8])*1e10*u.Msun
    distances = np.array([23,24,25])*u.kpc
    sizes = np.array([1,7,13])*u.kpc
    x = np.linspace(-0.18,-0.04,10000)
    
    for i, mw_label in enumerate(['td', 'halo']):
        for j, m in enumerate(masses):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d0.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
            
            f = h5py.File(fname, 'r')
            orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
            f.close()
            
            etot = orbit.energy()[-1]
            sigma_etot = 0.05
            kde = gaussian_kde(etot.value, bw_method=sigma_etot)
            y = kde(x)
            
            np.savez('../data/ehist_kde_{:s}'.format(root), x=x, y=y)
            
        for j, s in enumerate(sizes):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d0.to(u.kpc).value, (m0*1e-10).to(u.Msun).value, s.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
            
            f = h5py.File(fname, 'r')
            orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
            f.close()
            
            etot = orbit.energy()[-1]
            sigma_etot = 0.05
            kde = gaussian_kde(etot.value, bw_method=sigma_etot)
            y = kde(x)
            
            np.savez('../data/ehist_kde_{:s}'.format(root), x=x, y=y)
            
        for j, d in enumerate(distances):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m0*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
            
            f = h5py.File(fname, 'r')
            orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
            f.close()
            
            etot = orbit.energy()[-1]
            sigma_etot = 0.05
            kde = gaussian_kde(etot.value, bw_method=sigma_etot)
            y = kde(x)
            
            np.savez('../data/ehist_kde_{:s}'.format(root), x=x, y=y)

def ehist_suite():
    """"""
    
    # simulation setup
    Nskip = 10
    Nrand = 50000
    a = 7*u.kpc
    d0 = 24*u.kpc
    m0 = 1.4e10*u.Msun
    masses = np.array([0.7,1.4,2.8])*1e10*u.Msun
    distances = np.array([23,24,25])*u.kpc
    sizes = np.array([1,7,13])*u.kpc
    
    alphas = np.array([0.9,0.5,0.9])
    cmaps = [mpl.cm.Reds, mpl.cm.Blues]
    cmaps = [mpl.cm.autumn_r, mpl.cm.winter_r]
    
    x = np.linspace(-0.18,-0.04,10000)
    
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(12,9), sharex='col', sharey='row')
    
    for i, mw_label in enumerate(['td', 'halo']):
        for j, m in enumerate(masses):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d0.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/ehist_kde_{:s}.npz'.format(root)
            
            f = np.load(fname)
            x = f['x']
            y = f['y']
            
            plt.sca(ax[0][i])
            plt.plot(x, y, '-', color=cmaps[i]((j+1)/4), alpha=alphas[j], lw=2, zorder=1, label='$M_{{Sgr}}$ = {:.1e} $M_\odot$'.format(m.value))
            if j==1:
                plt.fill_between(x, y, color=cmaps[i]((j+1)/4), alpha=0.1, zorder=0)
        
        for j, s in enumerate(sizes):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d0.to(u.kpc).value, (m0*1e-10).to(u.Msun).value, s.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/ehist_kde_{:s}.npz'.format(root)
            
            f = np.load(fname)
            x = f['x']
            y = f['y']
            
            plt.sca(ax[1][i])
            plt.plot(x, y, '-', color=cmaps[i]((j+1)/4), alpha=alphas[j], lw=2, zorder=1, label='$a_{{Sgr}}$ = {:.0f} kpc'.format(s.value))
            if j==1:
                plt.fill_between(x, y, color=cmaps[i]((j+1)/4), alpha=0.1, zorder=0)
        
        for j, d in enumerate(distances):
            root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m0*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
            fname = '../data/ehist_kde_{:s}.npz'.format(root)
            
            f = np.load(fname)
            x = f['x']
            y = f['y']
            
            plt.sca(ax[2][i])
            plt.plot(x, y, '-', color=cmaps[i]((j+1)/4), alpha=alphas[j], lw=2, zorder=1, label='$d_{{Sgr}}$ = {:.0f} kpc'.format(d.value))
            if j==1:
                plt.fill_between(x, y, color=cmaps[i]((j+1)/4), alpha=0.1, zorder=0)
    
    coltitles = ['Disk', 'Halo']
    for i in range(2):
        plt.sca(ax[2,i])
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        
        plt.sca(ax[0,i])
        plt.title(coltitles[i], fontsize='medium')
    
    for i in range(3):
        plt.sca(ax[i,0])
        plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
        for j in range(2):
            plt.sca(ax[i,j])
            plt.legend(fontsize='small', handlelength=1, loc=1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ehist_model_comparison.png')



def plot_elz_sgr():
    """"""
    
    # range of distances to compare
    distances = np.arange(24,28.1,1)*u.kpc
    distances = np.arange(27,27.1,1)*u.kpc
    Ndist = np.size(distances)
    colors = [mpl.cm.inferno(x/Ndist) for x in range(Ndist)]
    
    # simulation setup
    m=1.4e10*u.Msun
    a=1*u.kpc
    Nr=300
    disk_label = 'logdisk'
    sigma_z = 0*u.kpc
    sigma_vz = 0*u.km/u.s
    mw_label = 'halo'
    
    # plotting setup
    eridge = np.array([-0.15106, -0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    eridge = np.array([1, 7/6., 4/3., 1.5, 1.8, 2])*-1
    
    fr = []
    for tr_ in np.abs(eridge):
        fr_ = Fraction('{:f}'.format(tr_)).limit_denominator(10)
        fr += [fr_]
        print(fr_, np.abs(1 - fr_.numerator/fr_.denominator/tr_))
    
    plt.close()
    plt.figure(figsize=(10,10))
    
    # plot different distances
    for i in range(Ndist):
        d = distances[i]
        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label)
        fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
        
        f = h5py.File(fname, 'r')
        orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
        f.close()
        
        energy = orbit.energy()
        epot = orbit.potential_energy()
        ekin = energy - epot
        l = orbit.angular_momentum()
        
        #w0 = orbit[-1][::10]
        #dt = 0.5*u.Myr
        #Tback = 2*u.Gyr
        #Nback = int((Tback/dt).decompose())
        
        #orbit_fixed = ham.integrate_orbit(w0, dt=-dt, n_steps=Nback)
        
        #energy = orbit_fixed.energy()
        #epot = orbit_fixed.potential_energy()
        #l = orbit_fixed.angular_momentum()
        
        peri, tperi = orbit[:,0].pericenter(func=None, return_times=True)
        ind_peri = np.argmin(np.abs(orbit.t - tperi[-1]))
        ecc = orbit[:,0].eccentricity()
        #print(ecc)
        
        label = '$d_{{Sgr}}$={:g}'.format(d.to(u.kpc))
        plt.plot(l.value[2,-1,1:] + 0*1.6*i, energy[-1,1:], '.', color=colors[i], ms=1, alpha=0.25)
        #plt.plot(l.value[2,0,1:] + 0*1.6*i, energy[0,1:], '.', color=colors[i], ms=1, alpha=0.25)
        #plt.plot(l.value[2,-1,1:] + 2*i, energy[-1,1:]/np.abs(energy[-1,0])*(0.5/ecc)**(1/3.), '.', color=colors[i], ms=1, alpha=0.25, label=label)
        #plt.plot(l.value[2,-1,1:] + 1.6*i, energy[-1,1:]/epot[ind_peri][0], '.', color=colors[i], ms=1, alpha=0.5)
        
    #for k, e in enumerate(eridge):
        #plt.axhline(e, lw=0.5, alpha=0.5)
        #fr_ = fr[k]
        #plt.text(7.5, e-0.05, '{:d}:{:d}'.format(fr_.numerator,fr_.denominator), ha='right', fontsize='small')
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.legend(loc=1, handlelength=0.5, markerscale=10, ncol=2, fontsize='small')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    #plt.ylabel('$E_{tot}$/|$E_{tot,Sgr}$| $\\times$ (0.5/$ecc_{Sgr}$)$^{1/3}$')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_sgr_{:s}.png'.format(root))

def plot_ehist_sgr():
    """"""
    # H3 data
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['eccen_pot1']>0.7)
    t = t[ind]
    
    # range of distances to compare
    distances = np.arange(24,28.1,1)*u.kpc
    distances = np.arange(27,27.1,1)*u.kpc
    Ndist = np.size(distances)
    colors = [mpl.cm.inferno(x/Ndist) for x in range(Ndist)]
    
    # simulation setup
    m=1.4e10*u.Msun
    a=1*u.kpc
    Nr=500
    disk_label = 'logdisk'
    sigma_z = 1*u.kpc
    sigma_vz = 0*u.km/u.s
    mw_label = 'halo'
    
    # plotting setup
    bins = np.linspace(-0.17,-0.04,90)
    eridge = np.array([-0.15106, -0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])

    plt.close()
    fig, ax = plt.subplots(Ndist, 1, figsize=(10,6), sharex=True)
    
    # plot different distances
    for i in range(Ndist):
        d = distances[i]
        root_disk = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
        fname_disk = '../data/sgr_hernquist_{:s}.h5'.format(root_disk)
        
        f_disk = h5py.File(fname_disk, 'r')
        orbit_disk = gd.Orbit(f_disk['pos'], f_disk['vel'], t=f_disk['time'], hamiltonian=ham, frame=gc_frame)
        f_disk.close()
        
        energy_disk = orbit_disk.energy()

        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label)
        fname = '../data/sgr_hernquist_{:s}.h5'.format(root)

        
        f = h5py.File(fname, 'r')
        orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
        f.close()
        
        energy = orbit.energy()
        lz = orbit.angular_momentum()[2][-1][1:]
        ind_prograde = lz<0
        ind_retrograde = lz>0
        
        print(np.shape(energy))
        Nt = np.shape(energy)[0]
        
        if Ndist>1:
            plt.sca(ax[i])

        plt.axvline(energy.value[-1,0], color='r')
        #plt.hist(energy_disk.value[-1,1:], bins=bins, histtype='step', color='r', label='', density=True)

        plt.hist(energy.value[-1,1:], bins=bins, histtype='step', color=colors[i], label='', density=True)
        #plt.hist(energy.value[-1,1:][ind_prograde], bins=bins, histtype='step', color='b', label='', density=True)
        #plt.hist(energy.value[-1,1:][ind_retrograde], bins=bins, histtype='step', color='dodgerblue', label='', density=True)
        
        #for i in range(Nt-1,0,-60):
            #plt.hist(energy.value[i,1:][ind_prograde], bins=bins, histtype='step', label='', density=True)
        
        #plt.hist(energy.value[0,1:][ind_prograde], bins=bins, histtype='step', color='tab:blue', label='', density=True)
        #plt.hist(energy.value[-1,1:][ind_retrograde], bins=bins, histtype='step', color='dodgerblue', label='', density=True)
        #plt.hist(energy.value[0,1:][ind_retrograde], bins=bins, histtype='step', color='skyblue', label='', density=True)
        #plt.hist(energy.value[0,1:], bins=bins, histtype='step', color='r', label='', density=True)
        
        plt.hist(t['E_tot_pot1'], bins=bins, histtype='step', color='k', label='H3 giants', density=True, alpha=0.3)

        #for e in eridge:
            #plt.axvline(e, lw=0.5, alpha=0.5)
        
        plt.ylabel('Density')
    
    plt.xlabel('Energy (km$^2$ s$^{-2}$)')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ehist_sgr_{:s}.png'.format(root))



def plot_elz(dg='lmc', Nback=3000):
    """"""
    f = h5py.File('../data/logcircular_plummer_{:s}_{:05d}.h5'.format(dg, Nback), 'r')
    
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    
    f.close()
    
    energy = orbit.energy()
    epot = orbit.potential_energy()
    l = orbit.angular_momentum()
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    t = t[ind]
    
    eridge = np.array([-0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    
    plt.close()
    plt.figure(figsize=(10,10))
    
    plt.plot(-t['Lz'], t['E_tot_pot1'], '.', color='navy', ms=2, alpha=0.5)
    plt.plot(l[2,-1,1:], energy[-1,1:], 'k.', ms=1, alpha=0.5)
    plt.plot(l[2,-1,0], energy[-1,0], 'r*', ms=5)
    plt.plot(l[2,-1,0], epot[-1,0], 'ro', ms=5)
    #plt.plot(l[2,0,1:], energy[0,1:], 'k.', ms=1, alpha=0.5)
    
    for e in eridge:
        plt.axhline(e, lw=0.5, alpha=0.5)
    
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_sgr_2.0.png')

def elz_movie(dg='lmc'):
    """"""
    f = h5py.File('../data/logcircular_plummer_{:s}.h5'.format(dg), 'r')
    
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    
    f.close()
    
    energy = orbit.energy()
    epot = orbit.potential_energy()
    l = orbit.angular_momentum()
    x = orbit.pos.x
    z = orbit.pos.z
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    t = t[ind]
    eridge = np.array([-0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    
    Nsnap = np.size(orbit.t)
    
    for i in range(Nsnap):
        plt.close()
        plt.figure(figsize=(10,10))
        
        plt.plot(-t['Lz'], t['E_tot_pot1'], '.', color='navy', ms=2, alpha=0.3, label='H3 giants')
        plt.plot(l[2,i,1:], energy[i,1:], 'k.', ms=1, alpha=0.1, label='Model (T={:.3f}Gyr)'.format(orbit.t[i]*1e-3))
        plt.plot(l[2,i,0], energy[i,0], 'r*', ms=5)
        plt.plot(l[2,i,0], epot[i,0], 'ro', ms=5)
        
        for e in eridge:
            plt.axhline(e, lw=0.5, alpha=0.2)
        
        plt.xlim(-6,6)
        plt.ylim(-0.18, -0.02)
        
        plt.legend(loc=2, markerscale=3, handlelength=0.7)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
        plt.tight_layout()
        
        iax = plt.gca().inset_axes([0.7,0.7,0.3,0.3])
        iax.plot(x[i,1:], z[i,1:], 'k.', ms=1, alpha=0.5)
        iax.plot(x[i,0], z[i,0], 'ro')
        iax.plot(x[:i+1,0], z[:i+1,0], 'r-', lw=0.5)
        
        iax.set_xlim(-100,100)
        iax.set_ylim(-100,100)
        iax.set_aspect('equal', adjustable='datalim')
        iax.set_axis_off()
        
        plt.savefig('../plots/elz_movie/logcircular/disk_sgr.{:03d}.png'.format(i))

def plot_ehist(dg='lmc', Nback=3000):
    """"""
    f = h5py.File('../data/logcircular_plummer_{:s}_{:05d}.h5'.format(dg, Nback), 'r')
    
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    
    f.close()
    
    energy = orbit.energy()
    l = orbit.angular_momentum()
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    t = t[ind]
    
    bins = np.linspace(-0.17,-0.04,100)
    eridge = np.array([-0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    
    plt.close()
    plt.figure(figsize=(15,7))
    
    plt.hist(t['E_tot_pot1'], bins=bins, histtype='step', color='navy', label='H3 giants', density=True, alpha=0.3)

    #plt.hist(energy.value[0,1:], bins=bins, histtype='step', color='r', label='Initial', density=True, alpha=0.1)
    plt.hist(energy.value[-1,1:], bins=bins, histtype='step', color='k', label='Final', density=True)
    
    for e in eridge:
        plt.axvline(e, lw=0.5, alpha=0.5)
    
    plt.legend()
    plt.xlabel('Energy (km$^2$ s$^{-2}$)')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('../plots/ehist_sgr_2.0.png')
    
def ehist_movie(dg='sgr', Nback=3000):
    """"""
    f = h5py.File('../data/logcircular_plummer_{:s}.h5'.format(dg), 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    energy = orbit.energy()
    x = orbit.pos.x
    z = orbit.pos.z
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    t = t[ind]
    
    Nsnap = np.size(orbit.t)
    bins = np.linspace(-0.17,-0.04,80)
    
    for i in range(Nsnap):
        plt.close()
        plt.figure(figsize=(15,7))
        
        plt.hist(t['E_tot_pot1'], bins=bins, histtype='step', color='k', label='H3 giants', density=True, alpha=0.3)

        #plt.hist(energy.value[0,1:], bins=bins, histtype='step', color='r', label='Initial', density=True, alpha=0.1)
        plt.hist(energy.value[i,1:], bins=bins, histtype='step', color='r', label='Model (T={:.3f}Gyr)'.format(orbit.t[i]*1e-3), density=True)
        
        plt.legend(loc=1)
        plt.xlabel('Energy (km$^2$ s$^{-2}$)')
        plt.ylabel('Density')
        
        plt.xlim(-0.17,-0.04)
        plt.ylim(0, 37)
        plt.tight_layout()
        
        iax = plt.gca().inset_axes([0.75,0.25,0.25,0.55])
        iax.plot(x[i,1:], z[i,1:], 'k.', ms=1, alpha=0.5)
        iax.plot(x[i,0], z[i,0], 'bo')
        iax.plot(x[:i+1,0], z[:i+1,0], 'b-')
        
        iax.set_xlim(-100,100)
        iax.set_ylim(-100,100)
        iax.set_aspect('equal', adjustable='datalim')
        iax.set_axis_off()
        
        plt.savefig('../plots/ehist_movie/logcircular/disk_sgr.{:03d}.png'.format(i))


def energy_time():
    """"""
    
    # simulation setup
    m=1.4e10*u.Msun
    a=1*u.kpc
    Nr=100
    disk_label = 'logdisk'
    sigma_z = 1*u.kpc
    sigma_vz = 0*u.km/u.s
    d = 27*u.kpc
    mw_label = 'halo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    energy = orbit.energy()
    epot = orbit.potential_energy()
    
    print(np.shape(energy))
    print(energy[:,0])
    
    t = orbit.t*1e-3
    Norb = np.shape(energy)[1]
    
    
    plt.close()
    plt.figure(figsize=(12,8))
    
    plt.plot(t, epot[:,0], 'k-', zorder=0)
    for i in range(1,Norb,2):
        #plt.plot(t, r[:,1::20], '-', lw=0.5, zorder=0)
        #e_ = energy[:,i]
        plt.plot(t, energy[:,i], '-', color=mpl.cm.viridis((energy.value[0,i]+0.16)/0.08), lw=0.2, alpha=0.3, zorder=1)
    
    #plt.gca().set_yscale('log')
    plt.xlim(0,np.max(t))
    plt.ylim(-0.17,-0.06)
    
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Energy [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/halo_energy_sgr_{:s}.png'.format(root))


#####
# Bar

def corot_func(r_cr, Omega, mw_pot):
    vc = mw_pot.circular_velocity([r_cr, 0., 0.])
    return abs(vc - Omega*r_cr * u.kpc).decompose().value[0]

def get_bar_model(Omega, Snlm, alpha=-27*u.deg, m=5e9*u.Msun, mw_pot=None):
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
        mw_pot = ham.potential

    res = minimize(corot_func, x0=4., args=(Omega, mw_pot))
    r_cr = res.x[0]
    r_s = r_cr / 3.67 # 3.67 scales this to the value WZ2012 use (60 km/s/kpc)

    return gp.SCFPotential(m=m / 10., r_s=r_s, # 10 is a MAGIC NUMBER: believe
                           Snlm=Snlm,
                           Tnlm=Snlm*0,
                           units=galactic,
                           R=rotation_matrix(alpha, 'z'))

def test_bar():
    """"""
    
    S = np.load('../data/Sn9l19m.npy') #expansion coeff.
    S = S[:6, :8, :8]
    
    xyz = np.zeros((3, 1024)) + 1e-8
    xyz[0] = np.linspace(0, 30, xyz.shape[1])

    ####Plot something using this loop
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.xlim(0, 30)
    plt.ylim(0, 300)

    Om = np.array([38,41,44])
    #Om = np.arange(38,45,1) #Array of pattern speeds, we are testing
    mw = ham.potential
    #mw =gp.BovyMWPotential2014()
    print(mw)
    print(mw['bulge'].parameters)

    for omega in Om*u.km/u.s/u.kpc:
        pot = gp.CCompositePotential()
        pot['disk'] = mw['disk']
        pot['halo'] = mw['halo']
        pot['bar'] = get_bar_model(Omega=omega, Snlm=S, m = 1e10)#, alpha=-27*u.deg) #set alpha 
        pot_m5e9 = gp.CCompositePotential()
        pot_m5e9['disk'] = mw['disk']
        pot_m5e9['halo'] = mw['halo']
        pot_m5e9['bar'] = get_bar_model(Omega=omega, Snlm=S, m = 5e9)
        #print(pot['bar'])
        
        plt.plot(xyz[0], pot.circular_velocity(xyz).to_value(u.km/u.s), color='steelblue', alpha=0.5)
        plt.plot(xyz[0], pot_m5e9.circular_velocity(xyz).to_value(u.km/u.s), color='maroon',alpha=0.5)
        plt.xlabel('Radius [kpc]')
        plt.ylabel('V$_c$ [km/s]')

    plt.axhspan(210, 230, zorder=-100, alpha=0.2, color='black')
    plt.axvline(8.1, zorder=-10, color='black')
    plt.plot(8.1,220, marker='$\odot$', color='black', markersize=20)
    
    plt.tight_layout()

def bar_orbits():
    """"""
    S = np.load('../data/Sn9l19m.npy') #expansion coeff.
    S = S[:6, :8, :8]
    omega = 41*u.km/u.s/u.kpc
    mw = ham.potential

    pot = gp.CCompositePotential()
    pot['disk'] = mw['disk']
    pot['halo'] = mw['halo']
    pot['bar'] = get_bar_model(Omega=omega, Snlm=S, m=1e10)

    frame = gp.ConstantRotatingFrame(Omega=[0,0,-1]*omega, units=galactic)
    H = gp.Hamiltonian(pot, frame) #frame such that we're "moving" with respect to bar
    
    Nrand = 1000
    seed = 193
    c_td = initialize_td(ret=True, Nrand=Nrand, seed=seed)
    seed = 191
    c_halo = initialize_halo(ret=True, Nrand=Nrand, seed=seed)
    c = coord.concatenate([c_td, c_halo])
    
    w0_mw = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    dt = 0.5*u.Myr
    rot_orbit = H.integrate_orbit(w0_mw, dt=dt, n_steps=5000, Integrator=gi.DOPRI853Integrator)
    
    orbit = gd.Orbit(rot_orbit.pos, rot_orbit.vel, t=rot_orbit.t, hamiltonian=ham, frame=gc_frame)
    
    
    plt.close()
    plt.figure()
    
    #etot = orbit.energy()[0].value
    #lz = orbit.angular_momentum()[2][0]
    #plt.plot(lz, etot, 'r.')
    
    etot = orbit.energy()[-1].value
    lz = orbit.angular_momentum()[2][-1]
    plt.plot(lz, etot, 'k.')
    
    plt.xlim(-6,6)
    plt.ylim(-0.2, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')

    plt.tight_layout()

def evolve_bar_stars(m=1e10*u.Msun, omega=41*u.km/u.s/u.kpc, T=3*u.Gyr, mw_label='td', Nrand=50000, seed=3928, Nskip=8, iskip=0, snap_skip=100, test=False):
    """
    Simulate bar perturbations starting from the present-day position
    m - bar mass (1e10 probably a tad more massive than the present-day bulge)
    omega - bar pattern speed (Sanders: 41 +/- 3 km/s/kpc)
    mw_label - [disk or halo] simulate H3-like population of disk or halo stars
    Nrand - number of stars to simulate
    seed - random seed
    Nskip - split particles over Nskip runs
    iskip - starting particle to start skipping
    snap_skip - number of timesteps to skip when saving orbits
    """
    
    # Set up bar
    S = np.load('../data/Sn9l19m.npy')
    S = S[:6, :8, :8]
    mw = ham.potential

    pot = gp.CCompositePotential()
    pot['disk'] = mw['disk']
    pot['halo'] = mw['halo']
    pot['bar'] = get_bar_model(Omega=omega, Snlm=S, m=m)

    #frame such that we're "moving" with respect to bar
    rot_frame = gp.ConstantRotatingFrame(Omega=[0,0,-1]*omega, units=galactic)
    
    ham_bar = gp.Hamiltonian(pot, rot_frame)

    # initialize star particles
    if mw_label=='halo':
        c = initialize_halo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)[iskip::Nskip]
    elif mw_label=='ndisk':
        c = initialize_ndisk(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='nhalo':
        c = initialize_nhalo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    else:
        mw_label = 'td'
        c = initialize_td(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    w0_mw = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    if test:
        print(c.x[0], np.size(c))
        return 0
    
    # integrate orbits
    dt = 0.5*u.Myr
    Nstep = int((T/dt).decompose())
    t1 = time.time()
    orbit = ham_bar.integrate_orbit(w0_mw, dt=dt, n_steps=Nstep, Integrator=gi.DOPRI853Integrator)
    t2 = time.time()
    print(t2-t1)
    
    # save orbits
    fname = '../data/bar_m.{:.1f}_om.{:2.0f}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(m.to(u.Msun).value*1e-10, omega.to(u.km/u.s/u.kpc).value, mw_label, Nrand, Nskip, iskip)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::snap_skip,:]
    orbit_out.to_hdf5(fout)
    fout.close()


# Configuration space

def plot_r(dg='lmc'):
    """"""
    f = h5py.File('../data/circular_plummer_{:s}.h5'.format(dg), 'r')
    
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    
    f.close()
    
    #energy = orbit.energy()
    #l = orbit.angular_momentum()
    r = orbit.cylindrical.rho
    
    rperi = orbit.pericenter()
    rapo = orbit.apocenter()
    r = 0.5 * (rperi + rapo)
    
    
    bins = np.linspace(0,40,200)
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.3)
    t = t[ind]
    
    rg = 0.5*(t['Rperi_pot1'] + t['Rapo_pot1'])
    
    plt.close()
    plt.figure(figsize=(15,7))
    
    plt.hist(r.value[0,1:], bins=bins, histtype='step', color='k', label='Initial', density=True, alpha=0.1)
    plt.hist(r.value[-1,1:], bins=bins, histtype='step', color='r', label='Final', density=True)

    #plt.hist(t['R_gal'], bins=bins, histtype='step', color='b', label='H3 giants', density=True)
    plt.hist(rg, bins=bins, histtype='step', color='tab:blue', label='H3 giants RG', density=True)
    
    plt.axvline(r[-1,0], color='orange', label='Sgr now')
    
    plt.legend()
    
    plt.tight_layout()

def disk_r():
    """"""
    # simulation setup
    m=1.4e10*u.Msun
    a=1*u.kpc
    Nr=100
    disk_label = 'logdisk'
    sigma_z = 1*u.kpc
    sigma_vz = 0*u.km/u.s
    d = 27*u.kpc
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    t = orbit.t*1e-3
    r = orbit.spherical.distance
    Norb = np.shape(r)[1]
    
    plt.close()
    plt.figure(figsize=(12,8))
    
    plt.plot(t, r[:,0], 'k-', zorder=0)
    for i in range(1,Norb,2):
        #plt.plot(t, r[:,1::20], '-', lw=0.5, zorder=0)
        r_ = r[:,i]
        plt.plot(t, r_, '-', color=mpl.cm.viridis((r_[0]-5)/35), lw=0.2, alpha=0.3, zorder=1)
    
    plt.gca().set_yscale('log')
    plt.xlim(0,np.max(t))
    plt.ylim(2,70)
    
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Galactocentric distance [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/disk_r_sgr_{:s}.png'.format(root))

def viz_disk_r():
    """"""
    # simulation setup
    m=1.4e10*u.Msun
    a=1*u.kpc
    Nr=300
    disk_label = 'logdisk'
    sigma_z = 1*u.kpc
    sigma_vz = 0*u.km/u.s
    d = 27*u.kpc
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    t = orbit.t*1e-3
    r = orbit.spherical.distance
    Norb = np.shape(r)[1]
    
    plt.close()
    plt.figure(figsize=(16,9), facecolor='k')
    plt.axes([0,0,1,1])
    
    #plt.plot(t, r[:,0], 'k-', zorder=0)
    plt.plot(t, r[:,0], '-', color='0.5', alpha=0.5, lw=0.5, zorder=0)
    
    for i in range(1,Norb,4):
        #plt.plot(t, r[:,1::20], '-', lw=0.5, zorder=0)
        r_ = r[:,i]
        plt.plot(t, r_, '-', color=mpl.cm.viridis((r_[0]-2)/38), lw=0.2, alpha=0.7, zorder=0)
    
    plt.gca().set_yscale('log')
    plt.xlim(0,np.max(t))
    plt.ylim(2,70)
    plt.axis('off')
    plt.gca().set_facecolor('k')
    
    #plt.xlabel('Time [Gyr]')
    #plt.ylabel('Galactocentric distance [kpc]')
    
    #plt.tight_layout()
    plt.savefig('../plots/viz_disk_r_sgr_{:s}.png'.format(root),dpi=100)


def periods():
    """"""
    
    # simulation setup
    m=0e10*u.Msun
    a=1*u.kpc
    Nr=300
    disk_label = 'logdisk'
    sigma_z = 1*u.kpc
    sigma_vz = 0*u.km/u.s
    #d = 24*u.kpc
    
    # range of distances to compare
    distances = np.arange(24,28.1,1)*u.kpc
    Ndist = np.size(distances)
    colors = [mpl.cm.Blues_r(x/Ndist) for x in range(Ndist)][::-1]
    
    # resonances
    eridge = np.array([1, 7/6., 4/3., 1.5, 1.8, 2, 9/4., 3, 4., 6.])
    eridge = np.array([2, 8/3., 9/4., 10/3., 4., 6.])
    eridge = np.array([1/2, 1/3., 1/4., 1/5., 1/6., 1/7.])
    
    # fractions
    fr = []
    for tr_ in np.abs(eridge):
        fr_ = Fraction('{:f}'.format(tr_)).limit_denominator(10)
        #fr_ = Fraction('{:f}'.format(tr_)).limit_numerator(10)
        fr += [fr_]
        #print(fr_, np.abs(1 - fr_.numerator/fr_.denominator/tr_))
    
    # H3
    t = Table.read('../data/rcat_giants.fits')
    ind_circ = (t['circLz_pot1']>0.35) & (t['Lz']<0)
    t = t[ind_circ]
    
    # APOGEE
    ta = Table(fits.getdata('../data/apogee_giants.fits'))
    ind = (ta['Lz']<-0.5) & (ta['dist']>2)
    ta = ta[ind]
    ind_high = (ta['zmax']>2)
    ind_low = (ta['zmax']<0.2)
    
    Tsgr = np.zeros(Ndist)
    bins = np.linspace(0,0.7,100)
    bins_apogee = np.linspace(0,0.7,200)
    #bins = np.linspace(0,500,70)
    #bins = np.logspace(np.log10(40), np.log10(500),75)
    #bins_h3 = np.logspace(np.log10(40), np.log10(500),75)
    
    plt.close()
    plt.figure(figsize=(15,6))
    
    
    for i in range(3,4):
        d = distances[i]
        root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_Nr.{:03d}_z.{:.1f}_vz.{:04.1f}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, disk_label, Nr, sigma_z.to(u.kpc).value, sigma_vz.to(u.km/u.s).value)
        fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
        
        f = h5py.File(fname, 'r')
        orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
        f.close()
        
        T = orbit.estimate_period()
        Tsgr[i] = T[0]
        print(T[0])

        
        label = '$d_{{Sgr}}$={:g}'.format(d.to(u.kpc))
        label = 'Model'
        plt.hist(T.value[1:]/T[0], bins=bins, color=colors[i], histtype='step', density=True, label=label, alpha=0.3, lw=2)
        
    #for k, e in enumerate(eridge):
        #plt.axvline(e, color='k', lw=0.5, alpha=0.2)
        #fr_ = fr[k]
        #plt.text(e+0.008, 6.5, '{:d}:{:d}'.format(fr_.numerator,fr_.denominator), va='top', fontsize='small', rotation=90, alpha=0.2)
    
    plt.hist(t['orbit_period_pot1']/T[0], bins=bins, color='tab:red', lw=1, alpha=0.5, histtype='step', density=True, label='H3')
    #plt.hist(ta['orbit_period_pot1'][ind_high]/T[0], bins=bins_apogee, color='green', lw=1, alpha=1, histtype='step', density=True, label='APOGEE, Zmax>2kpc')
    plt.hist(ta['orbit_period_pot1'][ind_low]/T[0], bins=bins_apogee, color='tab:orange', lw=1, alpha=0.5, histtype='step', density=True, label='APOGEE, Zmax<0.2kpc')
    
    plt.xlim(0.,0.7)
    plt.ylim(0.1,7)
    #plt.gca().set_yscale('log')
    plt.legend()
    plt.xlabel('T / T$_{Sgr}$')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('../plots/periods_sgr_{:s}.png'.format(root))


def sgr_pericenters(dg='sgr'):
    """"""
    f = h5py.File('../data/logcircular_plummer_{:s}.h5'.format(dg), 'r')
    
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    
    f.close()
    
    energy = orbit.energy()
    epot = orbit.potential_energy()
    ekin = orbit.kinetic_energy()
    l = orbit.angular_momentum()
    
    x = orbit.pos.x
    z = orbit.pos.z
    r = orbit.spherical.distance
    rho = orbit.cylindrical.rho

    t = orbit.t
    print(np.shape(r))
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(t, r[:,0], 'k-')
    plt.plot(t, z[:,0], 'r-')
    plt.plot(t, rho[:,0], 'b-')
    
    plt.axhline(0, color='k')
    
    plt.sca(ax[1])
    plt.plot(t, epot[:,0], 'r-')
    plt.plot(t, ekin[:,0], 'b-')
    
    
    timpact = [1195,2064,3107,3990]
    for ti in timpact:
        for i in range(2):
            plt.sca(ax[i])
            plt.axvline(ti)
    
    plt.tight_layout()

def vary_sgr():
    """Explore variations in Sgr orbit"""
    
    
    distances = np.arange(24,28.1,1)

    # orbit integration
    dt = 0.5*u.Myr
    T = 8*u.Gyr
    Nback = int((T/dt).decompose())
    
    
    plt.close()
    plt.figure(figsize=(12,8))
    
    for dist in distances:
        c = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=dist*u.kpc, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        
        orbit = ham.integrate_orbit(w0, dt=-dt, n_steps=Nback)
        
        label = 'r$_{{peri}}$={:.1f}, r$_{{apo}}$={:.1f}, T={:.1f}'.format(orbit.pericenter(), orbit.apocenter(), orbit.estimate_period().to(u.Gyr))
        
        plt.plot(orbit.t.to(u.Gyr), orbit.spherical.distance, '-', alpha=0.3,label=label)
        
        apo, tapo = orbit.apocenter(return_times=True, func=None)
        tbegin = tapo[-5]
        plt.axvline(tbegin.to(u.Gyr).value, color='k', lw=0.5)
        
        print(orbit.pericenter(), orbit.apocenter(), orbit.estimate_period(), tbegin)
        
        #plt.axhline(orbit.pericenter().value)
        #plt.axhline(orbit.apocenter().value)
        
    plt.legend()
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Galactocentric distance [kpc]')
    plt.tight_layout()




# general satellite

def satellite_orbits():
    """Determine a grid of initial satellite positions to cover the ELz space"""
    Nrow = 6
    Ncol = 4
    
    z = np.zeros((Nrow,Ncol)) * u.kpc
    y = np.zeros((Nrow,Ncol)) * u.kpc
    x = np.tile(np.logspace(1.45,2,Ncol), Nrow).reshape(Nrow, Ncol) * u.kpc
    
    q = np.array([x[0], y[0], z[0]])
    vcirc = ham.potential.circular_velocity(q)
    
    vx = np.zeros((Nrow,Ncol)) * u.km/u.s
    vy = np.tile(np.linspace(-1,1,Nrow), Ncol).reshape(Ncol, Nrow).T * vcirc[np.newaxis, :] * np.linspace(0.6,0.3,Ncol)[np.newaxis,:]
    vz = np.zeros((Nrow, Ncol)) * u.km/u.s
    
    c = coord.Galactocentric(x=x, y=y, z=z, v_x=vx, v_y=vy, v_z=vz)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    etot = orbit.energy()[0,:].value
    lz = orbit.angular_momentum()[2][0,:]
    
    # save initial positions
    pickle.dump(c, open('../data/satellite_init.pkl', 'wb'))
    
    # periods
    long_orbit = ham.integrate_orbit(w0, dt=1*u.Myr, n_steps=3000)
    print(long_orbit.estimate_period(radial=True))
    
    # H3 background
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3)
    t = t[ind]
    N = len(t)
    
    # weights
    ind_finite = (np.isfinite(t['E_tot_pot1_err'])) & (np.isfinite(t['Lz_err']))
    sigma_etot = (np.nanmedian(t['E_tot_pot1_err'][ind_finite])*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2).value
    sigma_lz = (np.nanmedian(t['Lz_err'][ind_finite])*u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1).value
    
    #2D histogram
    Nbin = 3000
    be_lz = np.linspace(-6, 6, Nbin)
    be_etot = np.linspace(-0.18, -0.02, Nbin)
    
    h, xe, ye = np.histogram2d(t['Lz'], t['E_tot_pot1'], bins=(be_lz, be_etot))
    h += 0.1
    
    detot = be_etot[1] - be_etot[0]
    dlz = be_lz[1] - be_lz[0]
    sigma_smooth = (sigma_etot/detot, sigma_lz/dlz)
    
    h_smooth = ndimage.gaussian_filter(h, sigma_smooth)
    
    plt.close()
    plt.figure()
    
    plt.imshow(h_smooth.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', norm=mpl.colors.LogNorm(), interpolation='none')
    plt.plot(lz, etot, 'ro')
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    

def evolve_satellite_stars(mw_label='idisk', Nrand=50000, seed=2874, isat=0, jsat=0, iskip=0, Nskip=8, snap_skip=100):
    """"""
    
    # initialize star particles
    if mw_label=='halo':
        c = initialize_halo(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    elif mw_label=='idisk':
        c = initialize_idisk(ret=True, Ntot=Nrand, Nr=1000, seed=seed)[iskip::Nskip]
    else:
        mw_label = 'td'
        c = initialize_td(ret=True, Nrand=Nrand, seed=seed)[iskip::Nskip]
    w0_mw = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = np.size(c.x)
    
    # initialize satellite
    sat_pot = gp.HernquistPotential(m=2e10*u.Msun, c=1*u.kpc, units=galactic)
    
    cs = pickle.load(open('../data/satellite_init.pkl', 'rb'))[isat][jsat]
    w0_sat = gd.PhaseSpacePosition(cs.transform_to(gc_frame).cartesian)
    
    # integrate forward
    w0 = gd.combine((w0_sat, w0_mw))
    particle_pot = [None] * (N + 1)
    particle_pot[0] = sat_pot
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    
    # determine period
    long_orbit = ham.integrate_orbit(w0_sat, dt=1*u.Myr, n_steps=3000)
    T = long_orbit.physicsspherical.estimate_period() * 2
    
    dt = 1*u.Myr
    #T = 3*u.Gyr
    Nfwd = int((T/dt).decompose())
    orbit = nbody.integrate_orbit(dt=dt, n_steps=Nfwd)
    
    # save orbits
    fname = '../data/satellite_{:d}.{:d}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(isat, jsat, mw_label, Nrand, Nskip, iskip)
    if os.path.exists(fname):
        os.remove(fname)
    
    fout = h5py.File(fname, 'w')
    orbit_out = orbit[::snap_skip,:]
    orbit_out.to_hdf5(fout)
    fout.close()

def run_satellites():
    """"""
    
    Nskip = 10
    for i in range(6):
        for j in range(4):
            for k in range(Nskip):
                print(i,j,k)
                evolve_satellite_stars(mw_label='idisk', Nrand=50000, seed=2874, isat=i, jsat=j, iskip=k, Nskip=Nskip, snap_skip=100)
    
    #for i in range(6):
        #for k in range(4):
            #print(i,k)
            #evolve_satellite_stars(mw_label='idisk', Nrand=5000, seed=2874, isat=i, jsat=k, iskip=0, Nskip=8, snap_skip=100)
    

def combine_satellite(isat=0, jsat=0):
    """"""
    # read in satellite
    Nrand = 50000
    Nskip = 10
    iskip = 0
    fname = '../data/satellite_{:d}.{:d}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(isat, jsat, mw_label, Nrand, Nskip, iskip)
    

def plot_elz_satellite(isat=0, jsat=0, mw_label='idisk'):
    """"""
    
    # read in satellite
    Nrand = 50000
    Nskip = 10
    iskip = 0
    fname = '../data/satellite_{:d}.{:d}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(isat, jsat, mw_label, Nrand, Nskip, iskip)
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    etot = orbit.energy()[-1,:]
    lz = orbit.angular_momentum()[2,-1,:]
    
    # H3 background
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3)
    t = t[ind]
    N = len(t)
    
    # weights
    ind_finite = (np.isfinite(t['E_tot_pot1_err'])) & (np.isfinite(t['Lz_err']))
    sigma_etot = (np.nanmedian(t['E_tot_pot1_err'][ind_finite])*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2).value
    sigma_lz = (np.nanmedian(t['Lz_err'][ind_finite])*u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1).value
    
    #2D histogram
    Nbin = 3000
    be_lz = np.linspace(-6, 6, Nbin)
    be_etot = np.linspace(-0.18, -0.02, Nbin)
    
    h, xe, ye = np.histogram2d(t['Lz'], t['E_tot_pot1'], bins=(be_lz, be_etot))
    h += 0.1
    
    detot = be_etot[1] - be_etot[0]
    dlz = be_lz[1] - be_lz[0]
    sigma_smooth = (sigma_etot/detot, sigma_lz/dlz)
    
    h_smooth = ndimage.gaussian_filter(h, sigma_smooth)
    
    plt.close()
    plt.figure()
    
    plt.imshow(h_smooth.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', norm=mpl.colors.LogNorm(), interpolation='none')
    plt.plot(lz[0], etot[0], 'wo', mew=0, ms=10, zorder=2)
    
    Ntot = 0
    for iskip in range(Nskip):
        fname = '../data/satellite_{:d}.{:d}_{:s}_N.{:06d}_{:d}.{:d}.h5'.format(isat, jsat, mw_label, Nrand, Nskip, iskip)
        f = h5py.File(fname, 'r')
        orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
        f.close()
        
        etot = orbit.energy()[-1,:]
        lz = orbit.angular_momentum()[2,-1,:]
        Ntot += np.size(etot)
        
        plt.plot(lz[1:], etot[1:], 'ro', mew=0, ms=0.5, zorder=1)
    
    #print(Ntot)
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_satellite_{:d}.{:d}.png'.format(isat, jsat))
