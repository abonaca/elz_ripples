import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.time import Time

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from superfreq import SuperFreq

#from galpy.orbit import Orbit
#from galpy.potential import MWPotential2014

from scipy.optimize import minimize, root
import functools
from scipy.stats import gaussian_kde

import time
import pickle
import h5py
from multiprocessing import Pool, Process
from fractions import Fraction

ham = gp.Hamiltonian(gp.MilkyWayPotential())

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

def bootstrap_elz(Nboot=1000, test=True):
    """"""
    
    ebins = np.linspace(-0.18,-0.05,150)
    snrs = np.array([5,10,20])
    #snrs = np.array([20,])
    Nsnr = np.size(snrs)
    
    egap = [-0.16397, -0.15766, -0.15249, -0.14117, -0.12969, -0.12403, -0.11891, -0.10511, -0.09767]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    for e in egap:
        for i in range(2):
            plt.sca(ax[i])
            plt.axvline(e, color='k', ls=':', lw=0.5)
    
    for i, snr in enumerate(snrs):
        t = Table.read('../data/rcat_giants.fits')
        ind = t['SNR']>snr
        t = t[ind]
        if test:
            t = t[:1000]
        N = len(t)
        
        ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0)
        ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.)
        
        # randomly draw 6D positions
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
        etot_med = np.median(etot, axis=1)
        etot_err = np.std(etot, axis=1)
        
        lz = orbit.angular_momentum()[2][0].reshape(N,-1)
        lz_med = np.median(lz, axis=1)
        lz_err = np.std(lz, axis=1)
        
        # plotting
        color_disk = mpl.cm.Reds((i+1)/Nsnr)
        color_halo = mpl.cm.Blues((i+1)/Nsnr)
        
        plt.sca(ax[0])
        plt.hist(etot[ind_disk].value.flatten(), bins=ebins, density=True, histtype='step', color=color_disk, label='S/N={:d} ({:d} stars)'.format(snr, np.sum(ind_disk)))
        #plt.hist(etot[ind_disk].value, bins=ebins, density=True, histtype='step', color=[color for x in range(Nboot)], alpha=0.05)
        #plt.hist(t['E_tot_pot1'][ind_disk], bins=ebins, density=True, histtype='step', color='k')
        
        plt.sca(ax[1])
        plt.hist(etot[ind_gse].value.flatten(), bins=ebins, density=True, histtype='step', color=color_halo, label='S/N={:d} ({:d} stars)'.format(snr, np.sum(ind_gse)))
        #plt.hist(etot[ind_gse].value, bins=ebins, density=True, histtype='step', color=[color for x in range(Nboot)], alpha=0.05)
        #plt.hist(t['E_tot_pot1'][ind_gse], bins=ebins, density=True, histtype='step', color='k')
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'Halo\n(ecc>0.7) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.legend(fontsize='small', bbox_to_anchor=(0.99,0.8), loc=1)
    
    plt.sca(ax[0])
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'Disk\n(circ>0.3) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.legend(fontsize='small', bbox_to_anchor=(0.99,0.8), loc=1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ehist_bootstrap.png')


def chemistry():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>18
    t = t[ind]
    N = len(t)
    age = 10**t['logAge']*1e-9
    
    # randomly draw 6D positions
    Nboot = 1000
    Ndim = 6
    seed = 927
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
    #lz = orbit.angular_momentum()[2][0].reshape(N,-1)
    
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    ind_halo = (t['eccen_pot1']>0.7) & (t['Lz']<0.)
    
    p_ge = [-0.32,-0.02]
    p_ge = [-0.14,0.17]
    poly_ge = np.poly1d(p_ge)

    p_splash = [-0.14,0.19]
    poly_splash = np.poly1d(p_splash)

    p_lowa = [-0.14,0.17]
    poly_lowa = np.poly1d(p_lowa)

    ind_lowachem = (t['init_FeH']>-0.65) & (t['init_aFe']<poly_lowa(t['init_FeH']))
    ind_mpoor = (t['init_FeH']<-0.7) & (t['init_aFe']<poly_ge(t['init_FeH']))
    ind_tdchem = (t['init_FeH']>-1) & (t['init_aFe']>poly_splash(t['init_FeH']))
    
    x_ge = np.linspace(-4, -0.7, 30)
    y_ge = poly_ge(x_ge)

    x_splash = np.linspace(-1, 0.5, 30)
    y_splash = poly_splash(x_splash)

    x_lowa = np.linspace(-0.65, 0.5, 30)
    y_lowa = poly_lowa(x_lowa)

    #print(np.sum(ind_disk & ind_lowachem))
    #print(np.sum(ind_disk & ind_tdchem))
    
    ebins = np.linspace(-0.18,-0.08,120)
    egap = [-0.16397, -0.15766, -0.15249, -0.14117, -0.12969, -0.12403, -0.11891, -0.10511, -0.09767]
    
    # plotting setup
    blue = '#0043c7'
    orange = '#db3700'
    gold = '#ffa006'
    lw = 2
    alpha = 0.6
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(12,6), sharex='col')
    
    plt.sca(ax[0,0])
    plt.plot(t['init_FeH'][ind_disk], t['init_aFe'][ind_disk], 'ko', alpha=0.3, mew=0, ms=3)
    plt.ylabel('[$\\alpha$/Fe]')
    plt.text(0.03,0.05, 'Disk\n(circ>0.3) & ($L_Z$<0)', va='bottom', ha='left', transform=plt.gca().transAxes, fontsize='small')
    
    plt.sca(ax[1,0])
    plt.plot(t['FeH'][ind_halo], t['aFe'][ind_halo], 'ko', alpha=0.3, mew=0, ms=3)
    plt.xlabel('[Fe/H]')
    plt.ylabel('[$\\alpha$/Fe]')
    plt.text(0.03,0.05, 'Halo\n(ecc>0.7) & ($L_Z$<0)', va='bottom', ha='left', transform=plt.gca().transAxes, fontsize='small')
    
    for i in range(2):
        plt.sca(ax[i,0])
        plt.plot(x_ge, y_ge, '-', color=blue, label='', lw=lw, alpha=alpha)
        plt.plot([-0.7, -0.7], [-0.2, poly_ge(-0.7)], '-', color=blue, label='', lw=lw, alpha=alpha)
        plt.plot(x_splash, y_splash, '-', color=orange, label='', lw=lw, alpha=alpha)
        plt.plot([-1, -1], [poly_splash(-1), 0.6], '-', color=orange, label='', lw=lw, alpha=alpha)
        plt.plot(x_lowa, y_lowa, '-', color=gold, label='', lw=lw, alpha=alpha)
        plt.plot([-0.65, -0.65], [-0.2, poly_lowa(-0.65)], '-', color=gold, label='', lw=lw, alpha=alpha)
        
        plt.xlim(-3.,0.)
        plt.ylim(-0.05,0.55)
    
    for e in egap:
        for i in range(2):
            plt.sca(ax[i,1])
            plt.axvline(e, color='k', ls=':', lw=0.5)
    
    plt.sca(ax[0,1])
    plt.hist(etot[ind_disk & ind_tdchem].value.flatten(), bins=ebins, histtype='step', color=orange, density=True, lw=lw, alpha=alpha)
    plt.hist(etot[ind_disk & ind_lowachem].value.flatten(), bins=ebins, histtype='step', color=gold, density=True, lw=lw, alpha=alpha)
    #plt.hist(etot[ind_disk & ind_mpoor].value.flatten(), bins=ebins, histtype='step', color=blue, density=True, lw=lw, alpha=alpha)
    plt.ylabel('Density [kpc$^{-2}$ Myr$^2$]')
    
    plt.sca(ax[1,1])
    plt.hist(etot[ind_halo & ind_tdchem].value.flatten(), bins=ebins, histtype='step', color=orange, density=True, lw=lw, alpha=alpha)
    plt.hist(etot[ind_halo & ind_mpoor].value.flatten(), bins=ebins, histtype='step', color=blue, density=True, lw=lw, alpha=alpha)
    plt.xlabel('$E_{tot}$ [kpc$^{2}$ Myr$^{-2}$]')
    plt.ylabel('Density [kpc$^{-2}$ Myr$^2$]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ehist_chemistry.png')


def compare_periods():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) # & (t['FeH']<-1)
    
    # gala estimate period close to radial period
    freq = 2*np.pi*t['orbit_period_pot1']**-1
    
    omegas = np.abs(np.array([t['omega_R'], t['omega_phi'], t['omega_z']]))
    residuals = 1 - omegas/freq[np.newaxis,:]
    print(np.nanmedian(residuals, axis=1))
    
    plt.close()
    plt.figure()
    
    plt.plot(freq, np.abs(t['omega_phi']), 'o')
    plt.plot(freq, np.abs(t['omega_z']), 'o')
    plt.plot(freq, np.abs(t['omega_R']), 'o')
    
    plt.plot(freq, freq, 'k-')
    
    plt.tight_layout()

def bootstrap_periods(test=True, snr=20, Nboot=1000):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>snr
    t = t[ind]
    print(len(t))
    if test:
        t = t[:10]
    N = len(t)
    
    # randomly draw 6D positions
    Ndim = 6
    seed = 251
    np.random.seed(seed)
    offsets = np.random.randn(N,Nboot,Ndim)
    
    ra = (t['GAIAEDR3_RA'][:,np.newaxis] + offsets[:,:,0] * t['GAIAEDR3_RA_ERROR'][:,np.newaxis]) * u.deg
    dec = (t['GAIAEDR3_DEC'][:,np.newaxis] + offsets[:,:,1] * t['GAIAEDR3_DEC_ERROR'][:,np.newaxis]) * u.deg
    dist = (t['dist_adpt'][:,np.newaxis] + offsets[:,:,2] * t['dist_adpt_err'][:,np.newaxis]) * u.kpc
    dist[dist<0*u.kpc] = 0*u.kpc
    pmra = (t['GAIAEDR3_PMRA'][:,np.newaxis] + offsets[:,:,3] * t['GAIAEDR3_PMRA_ERROR'][:,np.newaxis]) * u.mas/u.yr
    pmdec = (t['GAIAEDR3_PMDEC'][:,np.newaxis] + offsets[:,:,4] * t['GAIAEDR3_PMDEC_ERROR'][:,np.newaxis]) * u.mas/u.yr
    vr = (t['Vrad'][:,np.newaxis] + offsets[:,:,5] * t['Vrad_err'][:,np.newaxis]) * u.km/u.s
    
    # calculate orbital periods
    c = coord.SkyCoord(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    periods = np.empty((N,Nboot))*u.Myr
    t1 = time.time()
    for i in range(N):
        for j in range(Nboot):
            orbit = ham.integrate_orbit(w0[i,j], dt=1*u.Myr, n_steps=5000)
            periods[i,j] = orbit.estimate_period()
    t2 = time.time()
    print(t2-t1)
    
    pickle.dump(periods, open('../data/bootstrap_periods_snr.{:d}.pkl'.format(snr), 'wb'))
    
def best_distance(snr=20):
    """"""
    f = np.load('../data/sgr_freqs.npy')
    Pr_sgr = 2*np.pi/np.abs(f[:,0])
    d = np.linspace(24,28,500)*u.kpc
    print(np.argmin(np.abs(d-27.46*u.kpc)))
    
    N = np.size(Pr_sgr)
    chi_disk = np.zeros(N)
    chi_halo = np.zeros(N)
    chi_tot = np.zeros(N)
    
    #t = Table.read('../data/rcat_giants.fits')
    #Pr = 2*np.pi/np.abs(t['omega_R'])
    
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>snr
    t = t[ind]
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    ind_halo = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5)
    
    Pr = pickle.load(open('../data/bootstrap_periods_snr.{:d}.pkl'.format(snr),'rb'))
    
    
    epsilon = 0.1
    pres = np.array([1, 1/2., 1/3., 1/4., 1/5., 1/6., 1/7., 1/8., 1/9., 1/10.])
    
    for i in range(N):
        for p in pres:
            ind = (Pr.value > Pr_sgr[i]*p*(1-epsilon)) & (Pr.value < Pr_sgr[i]*p*(1+epsilon))
            ind2 = (Pr.value > Pr_sgr[i]*p*(1-2*epsilon)) & (Pr.value < Pr_sgr[i]*p*(1+2*epsilon))
            chi_disk[i] += np.sum(ind[ind_disk])/np.sum(ind2[ind_disk])
            chi_halo[i] += np.sum(ind[ind_halo])/np.sum(ind2[ind_halo])
            chi_tot[i] += np.sum(ind[ind_halo | ind_disk])/np.sum(ind2[ind_halo | ind_disk])
    
    plt.close()
    plt.figure()
    
    plt.plot(d, chi_disk, 'r-', alpha=0.2)
    plt.plot(d, chi_halo, 'b-', alpha=0.2)
    plt.plot(d, chi_tot, 'k-')
    
    plt.tight_layout()

def periods(snr=20):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>snr
    t = t[ind]
    
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) #& (t['SNR']>20)
    ind_halo = (t['eccen_pot1']>0.7) & (t['Lz']<0.) #& (t['SNR']>20) # & (t['FeH']<-1)
    
    # bootstrap periods
    P = pickle.load(open('../data/bootstrap_periods_snr.{:d}.pkl'.format(snr),'rb'))
    
    # Sgr period
    c_sgr = coord.ICRS(ra=283.76*u.deg, dec=-30.48*u.deg, distance=26.5*u.kpc, radial_velocity=142*u.km/u.s, pm_ra_cosdec=-2.7*u.mas/u.yr, pm_dec=-1.35*u.mas/u.yr)
    w0_sgr = gd.PhaseSpacePosition(c_sgr.transform_to(gc_frame).cartesian)
    
    # integrate orbit
    dt = 0.5*u.Myr
    Tfwd = 5*u.Gyr
    Nfwd = int((Tfwd/dt).decompose())
    
    sgr_orbit = ham.integrate_orbit(w0_sgr, dt=dt, n_steps=Nfwd)
    P_sgr = sgr_orbit.estimate_period()
    
    # bar
    omega_bar = np.array([38,41,44])*u.km/u.s/u.kpc
    P_bar = (2*np.pi/omega_bar).to(u.Myr)
    
    bins = np.linspace(60,1000,100)
    bins = np.logspace(np.log10(75), np.log10(1020), 200)
    
    cblue = mpl.cm.Blues(0.75)
    cred = mpl.cm.Reds(0.75)
    #cred = 'orangered'
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(20,7), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(P[ind_disk].value.flatten(), color=cred, histtype='step', density=True, bins=bins, alpha=0.5, lw=2)
    plt.ylabel('Density [Myr$^{-1}$]')
    plt.text(0.98,0.05, 'Disk\n(circ>0.3) & ($L_Z$<0)', va='bottom', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.ylim(0,0.0099)
    
    plt.sca(ax[1])
    plt.hist(P[ind_halo].value.flatten(), color=cblue, histtype='step', density=True, bins=bins, alpha=0.5, lw=2)
    plt.gca().set_xscale('log')
    
    plt.ylabel('Density [Myr$^{-1}$]')
    plt.xlabel('Orbital period [Myr]')
    plt.text(0.98,0.95, 'Halo\n(ecc>0.7) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    pres = np.array([1, 1/2., 1/3., 1/4., 1/5., 1/6., 1/7., 1/8., 1/9., 1/10.])
    Nres = np.size(pres)

    
    # draw resonance lines
    for i in range(2):
        plt.sca(ax[i])
        for res in pres:
            plt.axvline(res * P_sgr.value, color='k', lw=0.5, alpha=0.5)
        
        plt.axvspan(P_bar[0].value, P_bar[2].value, color='k', alpha=0.1, lw=0)
    
    fres = []
    for pr in pres:
        fr = Fraction('{:f}'.format(pr)).limit_denominator(14)
        fres += [fr]
    
    # resonance labels
    plt.sca(ax[0])
    ymax = 0.0099
    for i in range(Nres):
        l = 'P$_{{orb}}$:$P_{{Sgr,orb}}$ = {0:d}:{1:d}'.format(fres[i].denominator, fres[i].numerator)
        l = '{0:d}:{1:d}'.format(fres[i].denominator, fres[i].numerator)
        
        plt.text(0.99*pres[i]*P_sgr.value, 0.97*ymax, l, rotation=90, ha='right', va='top', fontsize='x-small')
    plt.text(1.01*P_sgr.value, 0.97*ymax, 'Sagittarius', rotation=90, ha='left', va='top', fontsize='x-small')
    
    plt.sca(ax[1])
    ymax = 0.0071
    plt.text(0.99*P_bar[0].value, 0.97*ymax, 'Bar corrotation', rotation=90, ha='right', va='top', fontsize='x-small')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/periods.png')
    #plt.savefig('../plots/narrative/periods.png')


def select_snapshots():
    """"""
    
    # simulation setup
    d = 24*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 50000
    mw_label = 'td'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    plt.close()
    plt.figure()
    
    plt.plot(orbit.t, orbit.spherical.distance[:,0], 'ko')
    
    plt.tight_layout()

def snapshots():
    """"""
    
    # simulation setup
    d = 24*u.kpc
    m = 1.4e10*u.Msun
    a = 1*u.kpc
    Nskip = 10
    Nrand = 50000
    mw_label = 'td'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit_disk = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    etot_disk = orbit_disk.energy()
    lz_disk = orbit_disk.angular_momentum()[2]
    
    mw_label = 'halo'
    
    root = 'd.{:.1f}_m.{:.1f}_a.{:02.0f}_{:s}_N.{:06d}_{:d}'.format(d.to(u.kpc).value, (m*1e-10).to(u.Msun).value, a.to(u.kpc).value, mw_label, Nrand, Nskip)
    fname = '../data/sgr_hernquist_{:s}.h5'.format(root)
    
    f = h5py.File(fname, 'r')
    orbit_halo = gd.Orbit(f['pos'], f['vel'], t=f['time'], hamiltonian=ham, frame=gc_frame)
    f.close()
    
    etot_halo = orbit_halo.energy()
    lz_halo = orbit_halo.angular_momentum()[2]
    
    ind_snap = [0,6,8,np.size(orbit_disk.t)-1]
    titles = ['Initial', 'First pericenter', '100 Myr past first pericenter', 'Final']
    cblue = mpl.cm.Blues(0.75)
    cred = mpl.cm.Reds(0.75)
    cred = 'orangered'
    cblue = 'dodgerblue'
    ebins = np.linspace(-0.18,-0.05,100)
    
    plt.close()
    fig, ax = plt.subplots(3,4,figsize=(14,12))
    
    for e, ind in enumerate(ind_snap):
        plt.sca(ax[0,e])
        plt.plot(orbit_halo.pos.x[ind,1::2], orbit_halo.pos.z[ind,1::2], 'o', color=cblue, ms=1, mew=0, alpha=0.3)
        plt.plot(orbit_disk.pos.x[ind,1::8], orbit_disk.pos.z[ind,1::8], 'o', color=cred, ms=1, mew=0, alpha=0.3)
        plt.plot(orbit_disk.pos.x[ind,0], orbit_disk.pos.z[ind,0], 'ko', ms=4)
        plt.plot(orbit_disk.pos.x[:ind+1,0], orbit_disk.pos.z[:ind+1,0], 'k-', alpha=0.2, lw=1)
        
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlim(-25,25)
        plt.ylim(-25,25)
        plt.xlabel('X [kpc]')
        plt.title(titles[e], fontsize='medium')
        
        plt.sca(ax[1,e])
        plt.plot(lz_halo[ind,1::10], etot_halo[ind,1::10], 'o', color=cblue, ms=1, mew=0, alpha=0.3, label='')
        plt.plot(lz_disk[ind,1::20], etot_disk[ind,1::20], 'o', color=cred, ms=1, mew=0, alpha=0.3, label='')
        if e>0:
            plt.plot(lz_disk[ind,0], etot_disk[ind,0], 'ko', ms=4, label='Sagittarius')
        
        plt.xlim(-6,6)
        plt.ylim(-0.2,-0.05)
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        
        plt.sca(ax[2,e])
        plt.hist(etot_halo.value[ind,1:], bins=ebins, color=cblue, histtype='step', alpha=0.3, lw=2)
        plt.hist(etot_disk.value[ind,1:], bins=ebins, color=cred, histtype='step', alpha=0.3, lw=2)
        if e>0:
            plt.axvline(etot_disk.value[ind,0], color='k', alpha=0.2)
        
        plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[0,0])
    plt.ylabel('Z [kpc]')
    
    plt.sca(ax[1,0])
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[2,0])
    plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
    plt.sca(ax[1,3])
    plt.plot(lz_halo[ind,1::10], -1*etot_halo[ind,1::10], 'o', color=cblue, ms=5, mew=0, alpha=0.3, label='Halo')
    plt.plot(lz_disk[ind,1::20], -1*etot_disk[ind,1::20], 'o', color=cred, ms=5, mew=0, alpha=0.3, label='Disk')
    plt.legend(fontsize='x-small', handlelength=0.4)
    
    for i in range(1,4):
        for j in range(3):
            plt.sca(ax[j,i])
            plt.gca().set_yticklabels([])
    
    plt.tight_layout(h_pad=0.2, w_pad=0.4)
    plt.savefig('../plots/model_snapshots.png')



# Auxiliary figures

def nobootstrap_elz(snr=20, test=False):
    """"""
    
    ebins = np.linspace(-0.18,-0.05,150)
    snrs = np.array([snr])
    #snrs = np.array([20,])
    Nsnr = np.size(snrs)
    
    egap = [-0.16397, -0.15766, -0.15249, -0.14117, -0.12969, -0.12403, -0.11891, -0.10511, -0.09767]
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    for e in egap:
        for i in range(2):
            plt.sca(ax[i])
            plt.axvline(e, color='k', ls=':', lw=0.5)
    
    for i, snr in enumerate(snrs):
        t = Table.read('../data/rcat_giants.fits')
        ind = t['SNR']>snr
        t = t[ind]
        if test:
            t = t[:1000]
        N = len(t)
        print(N)
        
        ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0)
        ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.)
        
        ## randomly draw 6D positions
        #Ndim = 6
        #seed = 1248
        #np.random.seed(seed)
        #offsets = np.random.randn(N,Nboot,Ndim)
        
        #ra = (t['GAIAEDR3_RA'][:,np.newaxis] + offsets[:,:,0] * t['GAIAEDR3_RA_ERROR'][:,np.newaxis]) * u.deg
        #dec = (t['GAIAEDR3_DEC'][:,np.newaxis] + offsets[:,:,1] * t['GAIAEDR3_DEC_ERROR'][:,np.newaxis]) * u.deg
        #dist = (t['dist_adpt'][:,np.newaxis] + offsets[:,:,2] * t['dist_adpt_err'][:,np.newaxis]) * u.kpc
        #dist[dist<0*u.kpc] = 0*u.kpc
        #pmra = (t['GAIAEDR3_PMRA'][:,np.newaxis] + offsets[:,:,3] * t['GAIAEDR3_PMRA_ERROR'][:,np.newaxis]) * u.mas/u.yr
        #pmdec = (t['GAIAEDR3_PMDEC'][:,np.newaxis] + offsets[:,:,4] * t['GAIAEDR3_PMDEC_ERROR'][:,np.newaxis]) * u.mas/u.yr
        #vr = (t['Vrad'][:,np.newaxis] + offsets[:,:,5] * t['Vrad_err'][:,np.newaxis]) * u.km/u.s
        
        ## calculate orbital properties
        #c = coord.SkyCoord(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
        #w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        #orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=0)
        
        #etot = orbit.energy()[0].reshape(N,-1)
        #etot_med = np.median(etot, axis=1)
        #etot_err = np.std(etot, axis=1)
        
        #lz = orbit.angular_momentum()[2][0].reshape(N,-1)
        #lz_med = np.median(lz, axis=1)
        #lz_err = np.std(lz, axis=1)
        
        # plotting
        color_disk = mpl.cm.Reds((i+1)/Nsnr)
        color_halo = mpl.cm.Blues((i+1)/Nsnr)
        
        plt.sca(ax[0])
        #plt.hist(etot[ind_disk].value.flatten(), bins=ebins, density=True, histtype='step', color=color_disk, label='S/N={:d} ({:d} stars)'.format(snr, np.sum(ind_disk)))
        #plt.hist(etot[ind_disk].value, bins=ebins, density=True, histtype='step', color=[color for x in range(Nboot)], alpha=0.05)
        plt.hist(t['E_tot_pot1'][ind_disk], bins=ebins, density=False, histtype='step', color=color_disk, label='S/N>{:d} ({:d} stars)'.format(snr, np.sum(ind_disk)))
        
        plt.sca(ax[1])
        #plt.hist(etot[ind_gse].value.flatten(), bins=ebins, density=True, histtype='step', color=color_halo, label='S/N={:d} ({:d} stars)'.format(snr, np.sum(ind_gse)))
        #plt.hist(etot[ind_gse].value, bins=ebins, density=True, histtype='step', color=[color for x in range(Nboot)], alpha=0.05)
        plt.hist(t['E_tot_pot1'][ind_gse], bins=ebins, density=False, histtype='step', color=color_halo, label='S/N>{:d} ({:d} stars)'.format(snr, np.sum(ind_gse)))
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('Number')
    plt.text(0.98,0.95, 'Halo\n(ecc>0.7) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.legend(fontsize='small', bbox_to_anchor=(0.99,0.8), loc=1)
    
    plt.sca(ax[0])
    plt.ylabel('Number')
    plt.text(0.98,0.95, 'Disk\n(circ>0.3) & ($L_Z$<0)', va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    plt.legend(fontsize='small', bbox_to_anchor=(0.99,0.8), loc=1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ehist_nobootstrap_snr{:d}.png'.format(snr))

