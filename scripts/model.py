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
import gala.integrate as gi

ham = gp.Hamiltonian(gp.MilkyWayPotential())

import scipy.stats
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy import ndimage

import pickle
import h5py

import os
from fractions import Fraction
import time

import readsnap

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()


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



##################
# Equilibrium disk

def nbody_disk():
    """"""
    filename = '../data/snap_010'
    
    posbulge = readsnap.read_block(filename, 'POS ', parttype=3) # load positions of halo particles
    velbulge = readsnap.read_block(filename, 'VEL ', parttype=3) # load positions of halo particles
    idbulge = readsnap.read_block(filename, 'ID  ', parttype=3) # load IDs of halo particles
    
    #poshalo = readsnap.read_block(filename, 'POS ', parttype=1) # load positions of halo particles
    #velhalo = readsnap.read_block(filename, 'VEL ', parttype=1) # load positions of halo particles
    #idhalo = readsnap.read_block(filename, 'ID  ', parttype=1) # load IDs of halo particles
    ##poshalo = poshalo[np.argsort(idhalo)]
    ##velhalo = velhalo[np.argsort(idhalo)]
    ##idhalo = np.sort(idhalo)
    
    #poshalo = poshalo - np.mean(posbulge, axis=0)
    #velhalo = velhalo - np.mean(velbulge, axis=0)
    
    posdisk = readsnap.read_block(filename, 'POS ', parttype=2) # load positions of disk particles
    veldisk = readsnap.read_block(filename, 'VEL ', parttype=2) # load positions of disk particles
    iddisk = readsnap.read_block(filename, 'ID  ', parttype=2) # load IDs of disk particles
    #posdisk = poshalo[np.argsort(iddisk)]
    #veldisk = velhalo[np.argsort(iddisk)]
    #iddisk = np.sort(iddisk)
    
    posdisk = posdisk - np.mean(posbulge, axis=0)
    veldisk = veldisk - np.mean(velbulge, axis=0)
    
    print(posdisk)
    print(veldisk)
    
    outdict = dict(x=posdisk, v=veldisk)
    pickle.dump(outdict, open('../data/thick_disk.pkl', 'wb'))

def plot_td():
    """"""
    td = pickle.load(open('../data/thick_disk.pkl', 'rb'))
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.plot(td['x'][:,0], td['x'][:,1], 'k.', mew=0, alpha=0.1, ms=2)
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
    
    c = coord.Galactocentric(x=td['x'][:,0], y=td['x'][:,1], z=td['x'][:,2], v_x=td['v'][:,0], v_y=td['v'][:,1], v_z=td['v'][:,2])
    np.random.seed(seed)
    
    
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
        plt.savefig('../plots/initialize_ndisk_phase.png')
    
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
    Nskip = 8
    distances = np.array([26.5,])
    masses = np.array([5,])*1e10
    sizes = np.array([2,])
    comp = ['idisk']

    for c in comp:
        for m in masses:
            for d in distances:
                for s in sizes:
                    print(c, m, d, s)
                    for i in range(Nskip):
                        t1 = time.time()
                        evolve_sgr_stars(m=m*u.Msun, d=d*u.kpc, a=s*u.kpc, Napo=4, Nrand=40000, mw_label=c, Nskip=Nskip, iskip=i, snap_skip=100)
                        t2 = time.time()
                        print(i, t2-t1)

def combine_skips():
    """"""
    # simulation setup
    d = 26.5*u.kpc
    m = 5e10*u.Msun
    a = 2*u.kpc
    Nskip = 8
    Nrand = 40000
    seed = 3928
    mw_label = 'idisk'
    
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


def diagnose_elz():
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
    print(np.size(e_init), np.size(np.unique(e_init)))
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    
    #t = Table.read('../data/rcat_giants.fits')
    #ind_gse = (t['eccen_pot1']>0.7) & (t['SNR']>5)
    #t = t[ind_gse]
    
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
    plt.plot(lz_fin, e_fin, 'ko', ms=ms, mew=0, alpha=alpha)
    plt.plot(lz_fin[0], e_fin[0], 'ro')
    
    #plt.xlim(-2,2)
    #plt.ylim(-0.18,-0.08)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_{:s}.png'.format(root))

def diagnose_ehist():
    """"""
    
    # simulation setup
    d = 27*u.kpc
    m = 1.4e10*u.Msun
    a = 7*u.kpc
    Nskip = 10
    Nrand = 50000
    mw_label = 'td'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    iskip = 0
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
    e_init = orbit.energy()[0]
    lz_init = orbit.angular_momentum()[2][0]
    
    e_fin = orbit.energy()[-1]
    lz_fin = orbit.angular_momentum()[2][-1]
    ind_pro = lz_fin.value<0
    ind_retro = lz_fin.value>0
    
    # H3
    t = Table.read('../data/rcat_giants.fits')
    if mw_label=='halo':
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
    bins = np.linspace(-0.16,-0.06,150)
    bins_coarse = np.linspace(-0.2,-0.02,50)
    bins_fine = np.linspace(-0.2,-0.02,200)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(15,10), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.hist(e_init.value, bins=bins, histtype='step', density=True, color='0.7', label='initial')
    plt.hist(e_fin.value, bins=bins, histtype='step', density=True, color='k', label='final')
    plt.hist(t['E_tot_pot1'], bins=bins, histtype='step', density=True, color='r', label='H3 giants')
    
    plt.legend(loc=2)
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    
    plt.hist(e_fin.value[ind_pro], bins=bins, histtype='step', density=True, color='tab:red', label='prograde')
    if mw_label=='halo':
        plt.hist(e_fin.value[ind_retro], bins=bins, histtype='step', density=True, color='tab:blue', label='retrograde')
    
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
    
    plt.legend(loc=2)
    plt.ylabel('Density')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/ehist_{:s}.png'.format(root))

    f.close()

def diagnose_periods():
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
    
    c = coord.Galactocentric(x=orbit.pos.x[-1]*u.kpc, y=orbit.pos.y[-1]*u.kpc, z=orbit.pos.z[-1]*u.kpc, v_x=orbit.vel.d_x[-1]*u.kpc/u.Myr, v_y=orbit.vel.d_y[-1]*u.kpc/u.Myr, v_z=orbit.vel.d_z[-1]*u.kpc/u.Myr)
    w0 = gd.PhaseSpacePosition(c.cartesian[::20])
    
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
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    bins = np.logspace(np.log10(75), np.log10(1020), 200)
    plt.hist(periods.value, bins=bins, histtype='step')
    
    for fr in [1, 2/3, 1/2, 2/7, 1/5, 1/8]:
        plt.axvline(periods.value[0]*fr)
    
    plt.gca().set_xscale('log')
    plt.xlabel('Period [Myr]')
    plt.ylabel('Number')
    plt.tight_layout()
    plt.savefig('../plots/period_{:s}.png'.format(root))
    
def aux_ratios():
    """"""
    
    a = np.array([947, 625, 467, 269, 185])
    print(a/a[0])


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
    orbit = ham_bar.integrate_orbit(w0_mw, dt=dt, n_steps=Nstep, Integrator=gi.DOPRI853Integrator)
    
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
