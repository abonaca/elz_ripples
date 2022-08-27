import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table, vstack #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.time import Time

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from superfreq import SuperFreq

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

from scipy.optimize import minimize, root
import functools
from scipy.stats import gaussian_kde
from scipy import ndimage
from skimage.filters import unsharp_mask
from scipy.fft import fft, ifft, fftfreq

import time
import pickle
from multiprocessing import Pool, Process
from fractions import Fraction

ham = gp.Hamiltonian(gp.MilkyWayPotential())

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()


def rcat_giants():
    """"""
    
    t = Table(fits.getdata('/home/ana/data/rcat.fits'))
    ind = (t['FLAG']==0) & (t['logg']<3.5) & (t['SNR']>3)
    t = t[ind]
    print(len(t), np.size(ind))
    
    t['Lx'] = (t['Lx']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Ly'] = (t['Ly']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lz'] = (t['Lz']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lperp'] = np.sqrt(t['Lx']**2 + t['Ly']**2)
    t['E_tot_pot1'] = (t['E_tot_pot1']*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2)
    
    t.write('../data/rcat_giants.fits', overwrite=True)

def rcat_history(v=1, tracer='giants'):
    """"""
    
    if v==1:
        fname = 'rcat_V3.0.3.d20201005_MSG.fits'
    elif v==2:
        fname = 'rcat_V4.0.3.d20201031_MSG.fits'
    else:
        v = 0
        fname = 'rcat_V4.1.5.d20220422_MSG.fits'
    
    t = Table(fits.getdata('/home/ana/data/h3/{:s}'.format(fname)))
    ind = (t['FLAG']==0) & (t['SNR']>3)
    if tracer=='giants':
        ind = ind & (t['logg']<3.5)
    elif tracer=='msto':
        ind = ind & (t['logg']<4.3) & (t['logg']>3.8)
    else:
        tracer = 'all'
    t.pprint()
    t = t[ind]
    print(len(t), np.size(ind))
    
    t['Lx'] = (t['Lx']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Ly'] = (t['Ly']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lz'] = (t['Lz']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lperp'] = np.sqrt(t['Lx']**2 + t['Ly']**2)
    t['E_tot_pot1'] = (t['E_tot_pot1']*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2)
    
    t.write('../data/rcat_{:s}_v{:d}.fits'.format(tracer, v), overwrite=True)

def rcat_all():
    """"""
    
    t = Table(fits.getdata('/home/ana/data/rcat.fits'))
    ind = (t['FLAG']==0) & (t['SNR']>3)
    t = t[ind]
    
    t['Lx'] = (t['Lx']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Ly'] = (t['Ly']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lz'] = (t['Lz']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lperp'] = np.sqrt(t['Lx']**2 + t['Ly']**2)
    t['E_tot_pot1'] = (t['E_tot_pot1']*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2)
    
    t.write('../data/rcat_all.fits', overwrite=True)

def rcat_msto():
    """"""
    t = Table(fits.getdata('/home/ana/data/rcat.fits'))
    ind = (t['FLAG']==0) & (t['logg']<4.3) & (t['logg']>3.8) & (t['SNR']>3)
    t = t[ind]
    
    t['Lx'] = (t['Lx']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Ly'] = (t['Ly']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lz'] = (t['Lz']*u.km/u.s*u.kpc).to(u.kpc**2/u.Myr)
    t['Lperp'] = np.sqrt(t['Lx']**2 + t['Ly']**2)
    t['E_tot_pot1'] = (t['E_tot_pot1']*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2)
    
    t.write('../data/rcat_msto.fits', overwrite=True)

def rcat_gse(tracer='stars', nskip=1, iskip=0):
    """"""
    
    t = Table.read('/home/ana/data/gse_{:s}.fits'.format(tracer))
    t = t[iskip::nskip]

    x0 = t['X'] * u.kpc
    y0 = t['Y'] * u.kpc
    z0 = t['Z'] * u.kpc
    
    vx0 = t['Vx'] * u.km/u.s
    vy0 = t['Vy'] * u.km/u.s
    vz0 = t['Vz'] * u.km/u.s
    
    ic_list = [x0, y0, z0, vx0, vy0, vz0]
    
    c = coord.Galactocentric(x=ic_list[0], y=ic_list[1], z=ic_list[2], v_x=ic_list[3], v_y=ic_list[4], v_z=ic_list[5])
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=2)
    
    if tracer=='stars':
        t.rename_column('Lz', 'Lz_sim')
    
    tout = t
    tout['E_tot_pot1'] = orbit.energy()[0]
    tout['Lx'] = orbit.angular_momentum()[0][0,:]
    tout['Ly'] = orbit.angular_momentum()[0][1,:]
    tout['Lz'] = orbit.angular_momentum()[0][2,:]
    
    tout.pprint()
    tout.write('../data/simcat_{:s}.fits'.format(tracer), overwrite=True)


def elz_errors(test=True, graph=True):
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    if test:
        t = t[:300]
    
    N = len(t)
    Nboot = 1000
    Ndim = 6
    seed = 381
    np.random.seed(seed)
    
    offsets = np.random.randn(N,Nboot,Ndim)
    
    ra = (t['GAIAEDR3_RA'][:,np.newaxis] + offsets[:,:,0] * t['GAIAEDR3_RA_ERROR'][:,np.newaxis]) * u.deg
    dec = (t['GAIAEDR3_DEC'][:,np.newaxis] + offsets[:,:,1] * t['GAIAEDR3_DEC_ERROR'][:,np.newaxis]) * u.deg
    dist = (t['dist_adpt'][:,np.newaxis] + offsets[:,:,2] * t['dist_adpt_err'][:,np.newaxis]) * u.kpc
    dist[dist<0*u.kpc] = 0*u.kpc
    pmra = (t['GAIAEDR3_PMRA'][:,np.newaxis] + offsets[:,:,3] * t['GAIAEDR3_PMRA_ERROR'][:,np.newaxis]) * u.mas/u.yr
    pmdec = (t['GAIAEDR3_PMDEC'][:,np.newaxis] + offsets[:,:,4] * t['GAIAEDR3_PMDEC_ERROR'][:,np.newaxis]) * u.mas/u.yr
    vr = (t['Vrad'][:,np.newaxis] + offsets[:,:,5] * t['Vrad_err'][:,np.newaxis]) * u.km/u.s
    
    c = coord.SkyCoord(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=0)
    
    etot = orbit.energy()[0].reshape(N,-1)
    etot_med = np.median(etot, axis=1)
    etot_err = np.std(etot, axis=1)
    
    lz = orbit.angular_momentum()[2][0].reshape(N,-1)
    lz_med = np.median(lz, axis=1)
    lz_err = np.std(lz, axis=1)
    
    if ~test:
        t['Lz_med'] = lz_med
        t['Lz_err'] = lz_err
        t['E_tot_pot1_med'] = etot_med
        t['E_tot_pot1_err'] = etot_err
        
        t.write('../data/rcat_giants.fits', overwrite=True)
    
    if graph:
        plt.close()
        fig, ax = plt.subplots(2,1,)
        
        plt.sca(ax[0])
        plt.plot(lz_med, 1 - lz_med/t['Lz'], 'ko', ms=3)
        
        plt.ylim(-0.1,0.1)
        
        plt.sca(ax[1])
        plt.plot(etot_med, 1 - etot_med/t['E_tot_pot1'], 'ko', ms=3)
        
        plt.tight_layout()

def galpy_actions(test=True, graph=True):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    if test:
        t = t[:300]
    
    c = coord.SkyCoord(ra=t['GAIAEDR3_RA']*u.deg, dec=t['GAIAEDR3_DEC']*u.deg, pm_ra_cosdec=t['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=t['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=t['Vrad']*u.km/u.s, distance=t['dist_adpt']*u.kpc, frame='icrs')

    o = Orbit(c)
    
    giants = dict()
    giants['Jr'] = (o.jr(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    giants['Jz'] = (o.jz(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    giants['Jphi'] = (o.jp(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    giants['Jtot'] = (giants['Jr']**2 + giants['Jz']**2 + giants['Jphi']**2)**0.5
    
    if ~test:
        for k in giants.keys():
            t[k] = giants[k]
            
            t.write('../data/rcat_giants.fits', overwrite=True)
    
    if graph:
        plt.close()
        plt.figure()
        plt.plot(giants['Jphi'], giants['Jr'], 'k.', ms=1, alpha=0.3)
        
        plt.xlim(-5,5)
        plt.ylim(0,5)
        plt.tight_layout()

def long_orbits(test=True, tracer='giants'):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    if test:
        t = t[:300]
    
    N = len(t)
    
    c = coord.SkyCoord(ra=t['GAIAEDR3_RA']*u.deg, dec=t['GAIAEDR3_DEC']*u.deg, pm_ra_cosdec=t['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=t['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=t['Vrad']*u.km/u.s, distance=t['dist_adpt']*u.kpc, frame='icrs')
    w = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    t1 = time.time()
    for i in range(N):
        w0 = w[i]
        orbit = ham.integrate_orbit(w0, dt=1*u.Myr, n_steps=100000)
        
        out = dict(t=orbit.t, w=orbit.w().T)
        pkl = pickle.dump(out, open('../data/long_orbits/{:s}.{:05d}.pkl'.format(tracer, i), 'wb'))
    t2 = time.time()
    
    tot = (t2 - t1)*u.s
    dt = tot/N
    print('{:f} {:f}'.format(tot.to(u.min), dt))

def long_orbits_sgr():
    """"""
    d = np.array([24,25,26,27,28])*u.kpc
    d = np.linspace(24,28,500)*u.kpc
    N = np.size(d)
    ra = np.array([283.76])*u.deg
    ra = np.repeat(ra, N)
    dec = np.array([-30.48])*u.deg
    dec = np.repeat(dec, N)
    vr = np.array([142])*u.km/u.s
    vr = np.repeat(vr, N)
    pmra = np.array([-2.7])*u.mas/u.yr
    pmra = np.repeat(pmra, N)
    pmdec = np.array([-1.35])*u.mas/u.yr
    pmdec = np.repeat(pmdec, N)
    
    
    c = coord.ICRS(ra=ra, dec=dec, distance=d, radial_velocity=vr, pm_ra_cosdec=pmra, pm_dec=pmdec)
    w = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    for i in range(N):
        w0 = w[i]
        orbit = ham.integrate_orbit(w0, dt=1*u.Myr, n_steps=100000)
        
        out = dict(t=orbit.t, w=orbit.w().T)
        pkl = pickle.dump(out, open('../data/long_orbits/sgr.{:05d}.pkl'.format(i), 'wb'))

def get_freq_sgr():
    """"""
    N = 500
    freqs = np.zeros((N, 3))
    
    for k in range(N):
        pkl = pickle.load(open('../data/long_orbits/sgr.{:05d}.pkl'.format(k), 'rb'))
        
        t = pkl['t']
        w = pkl['w']
        wp = gc.cartesian_to_poincare_polar(w)
        
        ndim = np.shape(wp)[1]
        
        sf = SuperFreq(t.value)
        fs = [(wp[:,i] * 1j*wp[:,i+ndim//2]) for i in range(ndim//2)]
        
        try:
            out = sf.find_fundamental_frequencies(fs)
            print(k, out.fund_freqs)
            freqs[k] = out.fund_freqs
        except:
            freqs[k] = np.nan
    
    np.save('../data/sgr_freqs', freqs)

def get_freq(i1, i2, verbose=False, tracer='giants'):
    """"""
    N = i2 - i1
    freqs = np.zeros((N, 3))
    
    for k in range(N):
        pkl = pickle.load(open('../data/long_orbits/{:s}.{:05d}.pkl'.format(tracer, k+i1), 'rb'))
        
        t = pkl['t']
        w = pkl['w']
        wp = gc.cartesian_to_poincare_polar(w)
        
        ndim = np.shape(wp)[1]
        
        sf = SuperFreq(t.value)
        fs = [(wp[:,i] * 1j*wp[:,i+ndim//2]) for i in range(ndim//2)]
        
        try:
            out = sf.find_fundamental_frequencies(fs)
            if verbose: print(i1+k, out.fund_freqs)
            freqs[k] = out.fund_freqs
        except:
            freqs[k] = np.nan
        
        np.save('../data/freqs/temp_{:s}.{:d}'.format(tracer, i1+k), freqs[k])
    
    np.save('../data/freqs_{:s}.{:d}.{:d}'.format(tracer, i1, i2), freqs)

def run_freqs_mp(nproc=8, test=True, tracer='giants'):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    if test:
        print(len(t))
        t = t[:100]
    
    Ntot = len(t)
    indices = np.linspace(0,Ntot,nproc+1, dtype='int')
    print(indices)
    for i in range(nproc):
        t_ = t[indices[i]:indices[i+1]]
        print(len(t_))
    
    processes = []

    t1 = time.time()
    for i in range(nproc):
        p = Process(target=get_freq, args=(indices[i], indices[i+1]), kwargs=dict(tracer=tracer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    t2 = time.time()
    print(t2-t1, (t2-t1)/Ntot)

def freqs(test=True):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    if test:
        t = t[:300]
    
    N = len(t)
    
    freqs = np.zeros((N, 3))
    
    t1 = time.time()
    for k in range(30):
        pkl = pickle.load(open('../data/long_orbits/giant.{:05d}.pkl'.format(k), 'rb'))
        
        t = pkl['t']
        w = pkl['w']
        wp = gc.cartesian_to_poincare_polar(w)
        
        ndim = np.shape(wp)[1]
        
        sf = SuperFreq(t.value)
        fs = [(wp[:,i] * 1j*wp[:,i+ndim//2]) for i in range(ndim//2)]
        
        try:
            out = sf.find_fundamental_frequencies(fs)
            freqs[k] = out.fund_freqs
        except:
            freqs[k] = np.nan
        
        if k%1000==0:
            np.save('../data/freqs.{:d}'.format(k), freqs)
            print(k)
    
    np.save('../data/freqs_all', freqs)
    t2 = time.time()
    
    tot = (t2 - t1)*u.s
    dt = tot/N
    print('{:f} {:f}'.format(tot.to(u.min), dt))

def store_freqs(save=False, test=False, tracer='giants'):
    """"""
    if test:
        Ntot = 100
        save = False
    else:
        t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
        Ntot = len(t)
    
    nproc = 8
    indices = np.linspace(0,Ntot,nproc+1, dtype='int')
    
    for i in range(nproc):
        d = np.load('../data/freqs_{:s}.{:d}.{:d}.npy'.format(tracer, indices[i], indices[i+1]))
        print(i, np.sum(np.isfinite(d[:,0])), np.shape(d), indices[i+1] - indices[i])
        
        if i==0:
            dall = d
        else:
            dall = np.vstack([dall,d])
    
    #print(dall[:,0], np.sum(np.isfinite(dall[:,0])))
    
    np.save('../data/freqs_{:s}'.format(tracer), dall)
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    
    trcat = t['orbit_period_pot1'][:Ntot]
    tr = 2*np.pi/dall[:,0]
    #print(trcat)
    #print(trcat/np.abs(tr))
    
    plt.close()
    plt.figure()
    plt.scatter(np.abs(dall[:,0]), np.abs(dall[:,1]), c=np.abs(t['Lz'][:Ntot].value), s=1)
    
    plt.tight_layout()

    if save:
        t['omega_R'] = dall[:,0] * u.Myr**-1
        t['omega_phi'] = dall[:,1] * u.Myr**-1
        t['omega_z'] = dall[:,2] * u.Myr**-1
        
        t['T_R'] = 2*np.pi * np.abs(dall[:,0])**-1 * u.Myr
        t['T_phi'] = 2*np.pi * np.abs(dall[:,1])**-1 * u.Myr
        t['T_z'] = 2*np.pi * np.abs(dall[:,2])**-1 * u.Myr
        
        
        t.pprint()
        t.write('../data/rcat_{:s}.fits'.format(tracer), overwrite=True)


def elz(snr=3):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>snr) & (t['zmax_pot1']>0) #& (t['eccen_pot1']>0.7)
    t = t[ind]
    print(len(t))
    print(np.nanmedian(t['E_tot_pot1_err']), np.nanmedian(t['Lz_err']), np.nanmedian(t['dist_adpt_err']/t['dist_adpt']))
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_giants_snr.{:d}.png'.format(snr), dpi=300)

def elz_chemistry(snr=10):
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>snr) #& (t['zmax_pot1']>0) #& (t['eccen_pot1']>0.7)
    t = t[ind]
    print(len(t))
    
    p_higha = [-0.14,0.18]
    poly_higha = np.poly1d(p_higha)
    ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
    
    p_lowa = [-0.14,0.15]
    poly_lowa = np.poly1d(p_lowa)
    ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    plt.plot(t['Lz'][ind_higha], t['E_tot_pot1'][ind_higha], 'ko', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.plot(t['Lz'][ind_lowa], t['E_tot_pot1'][ind_lowa], 'ko', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()

def elz_hist(snr=3, tracer='giants', weight=False):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    N = len(t)
    print(N)
    
    # weights
    ind_finite = (np.isfinite(t['E_tot_pot1_err'])) & (np.isfinite(t['Lz_err']))
    sigma_etot = (np.nanmedian(t['E_tot_pot1_err'][ind_finite])*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2).value
    sigma_lz = (np.nanmedian(t['Lz_err'][ind_finite])*u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1).value
    
    if weight:
        w = ((t['E_tot_pot1_err']/sigma_etot)**2 + (t['Lz_err']/sigma_lz)**2)**-0.5
    else:
        w = np.ones(N)
    
    #2D histogram
    Nbin = 3000
    be_lz = np.linspace(-6, 6, Nbin)
    be_etot = np.linspace(-0.18, -0.02, Nbin)
    
    h, xe, ye = np.histogram2d(t['Lz'], t['E_tot_pot1'], bins=(be_lz, be_etot), weights=w)
    h += 0.1
    
    detot = be_etot[1] - be_etot[0]
    dlz = be_lz[1] - be_lz[0]
    sigma_smooth = (sigma_etot/detot, sigma_lz/dlz)
    print(sigma_smooth)
    
    h_smooth = ndimage.gaussian_filter(h, sigma_smooth)
    
    h_sx = ndimage.sobel(h_smooth, axis=0, mode='constant')
    h_sy = ndimage.sobel(h_smooth, axis=1, mode='constant')
    h_edge = np.hypot(h_sx, h_sy)
    
    # colorbar scaling
    if tracer=='giants':
        vmax = 2e-3
    else:
        vmax = None
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(14,7), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.imshow(h_smooth.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', norm=mpl.colors.LogNorm(), interpolation='none')
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.imshow(h_edge.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', interpolation='none', cmap='binary', vmax=vmax)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    for i in range(2):
        plt.sca(ax[i])
        for e in eridge:
            plt.axhline(e, color='r', lw=0.2)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_hist_{:s}_w.{:d}.png'.format(tracer, weight))

def elz_unsharp(tracer='giants', snr=3):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    N = len(t)
    print(N)
    
    # weights
    ind_finite = (np.isfinite(t['E_tot_pot1_err'])) & (np.isfinite(t['Lz_err']))
    sigma_etot = (np.nanmedian(t['E_tot_pot1_err'][ind_finite])*u.km**2*u.s**-2).to(u.kpc**2*u.Myr**-2).value
    sigma_lz = (np.nanmedian(t['Lz_err'][ind_finite])*u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1).value
    
    #2D histogram
    Nbin = 1000
    be_lz = np.linspace(-6, 6, Nbin)
    be_etot = np.linspace(-0.18, -0.02, Nbin)
    
    h, xe, ye = np.histogram2d(t['Lz'], t['E_tot_pot1'], bins=(be_lz, be_etot))
    h += 0.1
    
    ## smooth by observational uncertainties
    #detot = be_etot[1] - be_etot[0]
    #dlz = be_lz[1] - be_lz[0]
    #sigma_smooth = (sigma_etot/detot, sigma_lz/dlz)
    #print(sigma_smooth)
    
    #h = ndimage.gaussian_filter(h, sigma_smooth)
    
    
    hu1 = unsharp_mask(h, radius=1, amount=1)
    hu5 = unsharp_mask(h, radius=5, amount=1)
    hu20 = unsharp_mask(h, radius=50, amount=1)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(12,12), sharex=True, sharey=True)
    
    plt.sca(ax[0,0])
    plt.imshow(h.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', norm=mpl.colors.LogNorm(), interpolation='none')
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[0,1])
    plt.imshow(hu1.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', interpolation='none', cmap='binary')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.sca(ax[1,0])
    plt.imshow(hu5.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', interpolation='none', cmap='binary')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1,1])
    plt.imshow(hu20.T, origin='lower', extent=(-6,6,-0.18,-0.02), aspect='auto', interpolation='none', cmap='binary')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    
    #eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    #for i in range(2):
        #for j in range(2):
            #plt.sca(ax[i][j])
            #for e in eridge:
                #plt.axhline(e, color='r', lw=0.2)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_unsharp_{:s}.png'.format(tracer))


def jrlz():
    """"""
    t = Table.read('../data/apogee_giants.fits')
    
    jr = np.sqrt(t['Jr'] / (8*u.kpc*220*u.km/u.s).to(u.kpc**2*u.Myr**-1))
    #jr = t['Jr'] / (8*u.kpc*220*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    lz = t['Lz'] / (-8*u.kpc*220*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    #print(jr)
    #print(t['Jr'])
    
    #print(np.sum(jr<10), len(t))
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    #plt.plot(t['Lz'], t['Jr'], 'ko', ms=2, mew=0, alpha=0.3)
    plt.plot(lz, jr, 'ko', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(0,2)
    plt.ylim(0,1)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$\sqrt{J_r}$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.tight_layout()

def elz_rgal():
    """"""
    N = 1000
    pos = np.zeros((3,N))*u.kpc
    pos[1] = np.linspace(1,100,N)*u.kpc
    
    mw = gp.MilkyWayPotential(halo={'m': 0.8*5.4e11*u.Msun}, disk={'m':0.1*6.8e10*u.Msun})
    mw = ham.potential

    vcirc = mw.circular_velocity(pos)
    vel = np.ones((3,N))*u.km/u.s
    vel[0] = vcirc
    
    etot = mw.total_energy(pos, vel)
    
    eridge = np.array([-0.15106, -0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    
    ind_close = [np.argmin(np.abs(etot - e*u.kpc**2*u.Myr**-2)) for e in eridge]
    r_ridge = np.array([pos[1][i].value for i in ind_close])
    
    t = Table.read('../data/rcat_giants.fits')
    ind_circ = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    emin = -0.2
    print(len(t))

    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(11.5,6), sharey=True, gridspec_kw=dict(width_ratios=(2,0.8,1)))
    
    plt.sca(ax[0])
    
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=1., mew=0, alpha=0.3, zorder=1)
    #plt.plot(t['Lz'][ind_circ], t['E_tot_pot1'][ind_circ], 'ko', ms=1.5, mew=0, alpha=0.15, zorder=1)
    
    #for e in eridge:
        #plt.axhline(e, color='tab:blue', lw=0.5, zorder=0, alpha=0.5)
    
    plt.xlim(-5,5)
    plt.ylim(-0.18, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    ebins = np.linspace(-0.18,-0.02,80)
    
    plt.hist(t['E_tot_pot1'][ind_circ], bins=ebins, orientation='horizontal', histtype='step', color='k')
    for e in eridge:
        plt.axhline(e, color='tab:blue', lw=0.5, zorder=0, alpha=0.5)
    
    #plt.xlabel('Number')
    plt.axis('off')
    
    plt.sca(ax[2])
    plt.plot(pos[1], etot, 'k-')
    
    plt.plot(r_ridge, eridge, 'o', ms=4, color='tab:blue')
    
    for e, r in zip(eridge, r_ridge):
        x = np.array([0, r, r])
        y = np.array([e, e, emin])
        plt.plot(x, y, '-', color='tab:blue', lw=0.5, alpha=0.5)
        
        plt.text(r+0.2, e-0.005, '{:.1f} kpc'.format(r), fontsize='x-small')

    plt.xlim(0,40)
    plt.xlabel('R$_{circular}$ [kpc]')
    
    
    plt.tight_layout(w_pad=0)
    plt.savefig('../plots/elz_rcirc.png')

def circularity():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>10) & (t['Lz']<0)
    t = t[ind]
    
    eridge = np.array([-0.1475, -0.1324, -0.1262, -0.1194, -0.1120, -0.0964])
    N = np.size(eridge)
    cridge = [mpl.cm.magma(x/N) for x in range(N)]
    climit = 0.4
    
    P1 = np.array([0.7814, -0.09327])
    P2 = np.array([0.6858, -0.09637])
    a = (P2[1] - P1[1])/(P2[0] - P1[0])
    b = P1[1] - a * P1[0]
    
    poly_ec = np.poly1d([a,b])
    x_ = np.linspace(0.5,0.9,100)
    y_ = poly_ec(x_)
    
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t['circLz_pot1'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.4)
    
    
    plt.sca(ax[1])
    plt.plot(t['circLz_pot1'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.4)
    plt.plot(x_, y_, 'r-')
    plt.plot(x_, y_ - 0.018, 'r-')
    plt.plot(x_, y_ - 0.024, 'r-')
    
    plt.ylim(-0.18,-0.05)
    plt.xlim(1,0)
    
    plt.tight_layout()


def disk_ridgeline(snr=3, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    
    # disk
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    
    Nbin = 30
    wbin = 0.002 * u.kpc**2*u.Myr**-2
    if tracer=='giants':
        ebins = np.linspace(-0.16, -0.08, Nbin) * u.kpc**2*u.Myr**-2
        pmin = 5
    else:
        ebins = np.linspace(-0.16, -0.1, Nbin) * u.kpc**2*u.Myr**-2
        pmin = 4
    
    lzmax = np.zeros(Nbin) #* u.kpc**2*u.Myr**-1
    for i in range(Nbin):
        ind_bin = (t['E_tot_pot1']>ebins[i] - wbin) & (t['E_tot_pot1']<ebins[i] + wbin)
        lzmax[i] = np.percentile(t['Lz'][ind_circular & ind_bin], pmin)
    
    if tracer=='giants':
        eridge = np.linspace(-0.16, -0.05, 100)
        k = 4
    else:
        eridge = np.linspace(-0.16, -0.1, 100)
        k = 4
    
    par = np.polyfit(ebins, lzmax, k)
    poly = np.poly1d(par)
    lzridge = poly(eridge)
    
    np.save('../data/elz_ridgeline_{:s}'.format(tracer), par)
    
    # halo
    ind_eccentric = (t['eccen_pot1']>0.7)
    
    lz_envelope = np.zeros((2,Nbin)) #* u.kpc**2*u.Myr**-1
    for i in range(Nbin):
        ind_bin = (t['E_tot_pot1']>ebins[i] - wbin) & (t['E_tot_pot1']<ebins[i] + wbin)
        lz_envelope[:,i] = np.percentile(t['Lz'][ind_eccentric & ind_bin],[30,70])
    
    par_prohalo = np.polyfit(ebins, lz_envelope[0], 4)
    poly_prohalo = np.poly1d(par_prohalo)
    lz_prohalo = poly_prohalo(eridge)

    par_rethalo = np.polyfit(ebins, lz_envelope[1], 4)
    poly_rethalo = np.poly1d(par_rethalo)
    lz_rethalo = poly_rethalo(eridge)
    
    np.save('../data/elz_envelope_{:s}'.format(tracer), [par_prohalo, par_rethalo])
    
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    #plt.plot(t['Lz'][ind_circular], t['E_tot_pot1'][ind_circular], 'ko', ms=2, mew=0, alpha=0.3)
    plt.plot(lzmax, ebins, 'ro')
    plt.plot(lzridge, eridge, 'r-')
    plt.plot(lzridge+0.3, eridge, 'r--')
    plt.plot(lzridge+1, eridge, 'r:')
    
    plt.plot(lz_prohalo, eridge, 'b-')
    plt.plot(lz_prohalo-0.3, eridge, 'b:')
    plt.plot(lz_rethalo, eridge, 'b-')
    plt.plot(lz_rethalo+0.3, eridge, 'b:')

    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_ridgeline_{:s}.png'.format(tracer))

def ridge_ehist(snr=3):
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    # Disk
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    
    par = np.load('../data/elz_ridgeline.npy')
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    # Halo
    ind_eccentric = (t['eccen_pot1']>0.7)
    
    par_prohalo, par_rethalo = np.load('../data/elz_envelope.npy')
    #d = np.load('../data/elz_envelope.npy')
    #print(d)
    
    poly_prohalo = np.poly1d(par_prohalo)
    poly_rethalo = np.poly1d(par_rethalo)
    
    ind_radial = (t['Lz'] > poly_prohalo(t['E_tot_pot1'])) & (t['Lz'] < poly_rethalo(t['E_tot_pot1']))
    ind_prohalo = (t['Lz'] > poly_prohalo(t['E_tot_pot1'])-0.3) & (t['Lz'] < poly_prohalo(t['E_tot_pot1']))
    ind_rethalo = (t['Lz'] > poly_rethalo(t['E_tot_pot1'])) & (t['Lz'] < poly_rethalo(t['E_tot_pot1'])+0.3)
    
    tgse = Table.read('../data/simcat_stars.fits')
    tdm = Table.read('../data/simcat_dm.fits')
    
    ebins = np.linspace(-0.18,-0.08,80)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(t['E_tot_pot1'][ind_circular & ind_ridge], bins=ebins, density=True, histtype='step', color='b', alpha=0.3)
    plt.hist(t['E_tot_pot1'][ind_circular & ind_ripple], bins=ebins, density=True, histtype='step', color='b')
    
    plt.sca(ax[1])
    plt.hist(t['E_tot_pot1'][ind_eccentric & (ind_radial | ind_prohalo)], bins=ebins, density=True, histtype='step', color='b')
    #plt.hist(t['E_tot_pot1'][ind_eccentric], bins=ebins, density=True, histtype='step', color='b', alpha=0.3)

    ebins = np.linspace(-0.18,-0.08,200)
    plt.hist(tgse['E_tot_pot1'], bins=ebins, density=True, histtype='step', color='r')
    #plt.hist(tdm['E_tot_pot1'], bins=ebins, density=True, histtype='step', color='k')
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115, -0.111, -0.106])*u.kpc**2*u.Myr**-2
    #elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    for j in range(Nlevel):
        for i in range(2):
            plt.sca(ax[i])
            plt.axvspan(elevels[2*j].value, elevels[2*j+1].value, color=mpl.cm.Oranges_r(j/(Nlevel+1)), alpha=0.1)
    
    plt.tight_layout()

def ridge_zhist(snr=3):
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    # Disk
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    
    par = np.load('../data/elz_ridgeline.npy')
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    ind_energy = [(t['E_tot_pot1']>elevels[2*x]) & (t['E_tot_pot1']<elevels[2*x+1]) for x in range(Nlevel)]
    
    cmaps = [mpl.cm.Reds, mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    
    zbins = np.linspace(1,10,100)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(t['Z_gal'][ind_circular & ind_ridge], bins=zbins, density=True, histtype='step', color='b', alpha=0.3)
    plt.hist(t['Z_gal'][ind_circular & ind_ripple], bins=zbins, density=True, histtype='step', color='b')
    
    plt.sca(ax[1])
    plt.hist(t['zmax_pot1'][ind_circular & ind_ridge], bins=zbins, density=True, histtype='step', color='b', alpha=0.3)
    plt.hist(t['zmax_pot1'][ind_circular & ind_ripple], bins=zbins, density=True, histtype='step', color='b')
    

    zbins = np.linspace(1,10,60)
    for i in range(1,Nlevel-1):
        #ind = ind_energy[i] & ind_ridge & ind_circular
        #plt.hist(t['zmax_pot1'][ind], bins=zbins, density=True, histtype='stepfilled', alpha=0.2, color=cmaps[i](0.5))
        
        ind = ind_energy[i] & ind_ripple & ind_circular
        plt.hist(t['zmax_pot1'][ind], bins=zbins, density=True, histtype='stepfilled', alpha=0.2, color=cmaps[i](0.85))
    
    plt.tight_layout()


def ripple_chemistry(snr=10, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    #t = Table.read('../data/rcat_msto.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    de = np.abs(elevels[::2] - elevels[1::2])
    print(np.sqrt(de).to(u.km/u.s))
    
    ind_energy = [(t['E_tot_pot1']>elevels[2*x]) & (t['E_tot_pot1']<elevels[2*x+1]) for x in range(Nlevel)]
    
    #ind_e1 = (t['E_tot_pot1']>elevels[0]) & (t['E_tot_pot1']<elevels[1])
    #ind_e2 = (t['E_tot_pot1']>elevels[2]) & (t['E_tot_pot1']<elevels[3])
    #ind_e3 = (t['E_tot_pot1']>elevels[4]) & (t['E_tot_pot1']<elevels[5])
    #ind_e4 = (t['E_tot_pot1']>elevels[6]) & (t['E_tot_pot1']<elevels[7])
    #ind_e5 = (t['E_tot_pot1']>elevels[8]) & (t['E_tot_pot1']<elevels[9])
    
    #ind_energy = [ind_e1, ind_e2, ind_e3, ind_e4, ind_e5]
    #cmaps = [mpl.cm.OrRd, mpl.cm.YlGn, mpl.cm.PuBu, mpl.cm.BuPu, mpl.cm.Blues, mpl.cm.Oranges]
    cmaps = [mpl.cm.Reds, mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    
    plt.close()
    fig = plt.figure(figsize=(15,8))
    
    gs1 = mpl.gridspec.GridSpec(1,1)
    gs1.update(left=0.08, right=0.45, top=0.95, bottom=0.1, hspace=0.05)

    gs2 = mpl.gridspec.GridSpec(Nlevel,1)
    gs2.update(left=0.52, right=0.975, top=0.95, bottom=0.1, hspace=0.05)

    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs2[0])
    ax2 = fig.add_subplot(gs2[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs2[2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs2[3], sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(gs2[4], sharex=ax1, sharey=ax1)
    #ax6 = fig.add_subplot(gs2[5], sharex=ax1, sharey=ax1)
    
    ax = [ax0, ax1, ax2, ax3, ax4, ax5]
    
    # plot background
    plt.sca(ax[0])
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    
    plt.xlim(-5,5)
    plt.ylim(-0.18, -0.04)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    for i in range(Nlevel):
        plt.sca(ax[i+1])
        plt.plot(t['init_FeH'], t['init_aFe'], 'ko', ms=2, mew=0, alpha=0.2)
        #plt.plot(t['FeH'], t['aFe'], 'ko', ms=2, mew=0, alpha=0.2)

        plt.xlim(-2.8,0.3)
        plt.ylim(-0.05,0.55)
        plt.ylabel('[$\\alpha/Fe$]$_{init}$')
    plt.xlabel('[Fe/H]$_{init}$')
    
    ms = 6
    
    for i in range(Nlevel):
        ind = ind_energy[i] & ind_ridge & ind_circular
        plt.sca(ax[0])
        plt.plot(t['Lz'][ind], t['E_tot_pot1'][ind], 'o', ms=2, mew=0, c=cmaps[i](0.5))
        
        plt.sca(ax[Nlevel-i])
        plt.plot(t['init_FeH'][ind], t['init_aFe'][ind], 'o', ms=ms, mew=0, c=cmaps[i](0.5))
        
        ind = ind_energy[i] & ind_ripple & ind_circular
        plt.sca(ax[0])
        plt.plot(t['Lz'][ind], t['E_tot_pot1'][ind], 'o', ms=2, mew=0, c=cmaps[i](0.85))
        
        plt.sca(ax[Nlevel-i])
        plt.plot(t['init_FeH'][ind], t['init_aFe'][ind], 'o', ms=ms, mew=0, c=cmaps[i](0.85))
    
    #plt.tight_layout()
    plt.savefig('../plots/ripples_chemistry_{:s}.png'.format(tracer))

def ripple_age(snr=10):
    """"""
    
    #t = Table.read('../data/rcat_giants.fits')
    t = Table.read('../data/rcat_msto_v0.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    p_higha = [-0.14,0.18]
    poly_higha = np.poly1d(p_higha)
    ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
    
    p_lowa = [-0.14,0.15]
    poly_lowa = np.poly1d(p_lowa)
    ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['E_tot_pot1']>-0.15) & (t['E_tot_pot1']<-0.115)
    
    par = np.load('../data/elz_ridgeline_msto.npy')
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    # Halo
    ind_eccentric = (t['eccen_pot1']>0.7) & (t['E_tot_pot1']>-0.15) & (t['E_tot_pot1']<-0.115)
    p_gse = [-0.32,-0.02]
    poly_gse = np.poly1d(p_gse)
    ind_gse = (t['init_FeH']<-0.6) & (t['init_aFe']<poly_gse(t['init_FeH']))
    
    
    age = 10**t['logAge']*1e-9
    age_lerr = age - 10**(t['logAge'] - t['logAge_lerr'])*1e-9
    age_uerr = 10**(t['logAge'] + t['logAge_uerr'])*1e-9 - age
    age_err = 0.5 * (age_lerr + age_uerr)
    
    abins = np.linspace(4,14,25)
    
    plt.close()
    plt.figure(figsize=(10,7))
    
    plt.hist(age[ind_circular & ind_ridge & ind_higha], bins=abins, density=True, histtype='step', color='r')
    plt.hist(age[ind_circular & ind_ripple & ind_higha], bins=abins, density=True, histtype='step', color='r', ls=':')

    plt.hist(age[ind_circular & ind_ridge & ind_lowa], bins=abins, density=True, histtype='step', color='gold')
    plt.hist(age[ind_circular & ind_ripple & ind_lowa], bins=abins, density=True, histtype='step', color='gold', ls=':')
    
    print(np.sum(ind_circular & ind_ripple & ind_lowa)/np.sum(ind_circular & ind_ridge & ind_lowa))
    #print(np.sum(ind_circular & ind_ridge & ind_lowa))
    
    print(np.sum(ind_circular & ind_ripple & ind_higha)/np.sum(ind_circular & ind_ridge & ind_higha))
    #print(np.sum(ind_circular & ind_ridge & ind_higha))
    


    plt.hist(age[ind_eccentric & ind_gse], bins=abins, density=True, histtype='step', color='b')
    
    plt.xlabel('Age [Gyr]')
    plt.ylabel('Density [Gyr$^{-1}$]')
    #tg = Table.read('../data/rcat_giants.fits')
    #ind_sort = np.argsort(age)[::-1]
    #plt.scatter(t['Lz'][ind_sort], t['E_tot_pot1'][ind_sort], c=age[ind_sort], s=5)

    #plt.ylim(-0.18,-0.1)
    plt.tight_layout()

def ripple_rguide(tracer='msto'):
    """"""
    if tracer=='msto':
        snr = 20
    else:
        snr = 3
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    ind_circular = (t['circLz_pot1']>0.5) & (t['Lz']<0)
    t = t[ind & ind_circular]
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    rbins = np.linspace(3,13,50)
    
    rg = 0.5 * (t['Rapo_pot1'] + t['Rperi_pot1'])
    
    rbins = np.logspace(0.3, 1.5, 100)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(7,7), sharex=True)
    
    plt.sca(ax[0])
    #plt.hist(rg, bins=rbins, histtype='step', color='k', alpha=0.2)
    plt.hist(rg[ind_ridge], bins=rbins, histtype='step', color='k', density=True, alpha=0.5)
    plt.hist(rg[ind_ripple], bins=rbins, histtype='step', color='orangered', density=True)
    
    plt.ylabel('Number')
    
    plt.sca(ax[1])
    plt.plot(rg[ind_ridge], t['zmax_pot1'][ind_ridge], 'k.', ms=2, alpha=0.2)
    plt.plot(rg[ind_ripple], t['zmax_pot1'][ind_ripple], 'o', ms=2, mew=0, color='orangered')
    #plt.hist(rg[ind_ripple], bins=rbins, histtype='step', density=True, color='b', label='giants')
    #plt.hist(rg_msto[ind_ripple_msto], bins=rbins, histtype='step', density=True, color='r', label='MSTO')
    
    plt.xlim(4,40)
    plt.xlabel('$R_{guide}$ [kpc]')
    plt.ylabel('$z_{max}$ [kpc]')
    #plt.legend(loc=1, fontsize='small')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    #plt.gca().set_aspect('equal')
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/ripple_rguide_hist_{:s}.png'.format(tracer), dpi=300)

def ripple_period(snr=10):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) #& (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.7)
    ind_sgr = (t['Sgr_FLAG']==1)
    
    tracer = 'giants'
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    de = np.abs(elevels[::2] - elevels[1::2])
    #print(np.sqrt(de).to(u.km/u.s))
    
    ind_energy = [(t['E_tot_pot1']>elevels[2*x]) & (t['E_tot_pot1']<elevels[2*x+1]) for x in range(Nlevel)]
    cmaps = [mpl.cm.Reds, mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    
    pbins = np.linspace(70,600,150)
    pbins_coarse = np.linspace(70,600,70)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(t['orbit_period_pot1'][ind_circular], bins=pbins, histtype='step', density=True, color='r', label='disk')
    plt.hist(t['orbit_period_pot1'][ind_gse], bins=pbins, histtype='step', density=True, color='navy', label='GSE')
    plt.hist(t['orbit_period_pot1'][ind_sgr], bins=pbins_coarse, histtype='step', density=True, color='orange', label='Sgr')
    plt.legend(loc=9, fontsize='small')
    plt.ylabel('Density [Myr$^{-1}$]')
    
    plt.sca(ax[1])
    plt.hist(t['orbit_period_pot1'][ind_circular], bins=pbins, histtype='step', density=False, color='r', label='disk')
    for i in range(Nlevel):
        plt.hist(t['orbit_period_pot1'][ind_ripple & ind_energy[i]], bins=pbins, histtype='step', density=False, color=cmaps[i](0.85), label='')
        plt.hist(t['orbit_period_pot1'][ind_ridge & ind_energy[i]], bins=pbins, histtype='step', density=False, color=cmaps[i](0.5), label='')
    
    plt.xlabel('Period [Myr]')
    plt.ylabel('Density [Myr$^{-1}$]')
    
    plt.tight_layout()
    #plt.savefig('../plots/ripple_period.png')

def ripple_vrgal():
    """"""
    
    tracer = 'giants'
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>10)
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
    t = t[ind & ind_circular]
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    
    ind_energy = [(t['E_tot_pot1']>elevels[2*x]) & (t['E_tot_pot1']<elevels[2*x+1]) for x in range(Nlevel)]
    
    for i in range(Nlevel):
        print(i)
        print(np.median(t['Vr_gal'][ind_energy[i] & ind_ridge]), np.std(t['Vr_gal'][ind_energy[i] & ind_ridge]))
        print(np.median(t['Vr_gal'][ind_energy[i] & ind_ripple]), np.std(t['Vr_gal'][ind_energy[i] & ind_ripple]))

def chemistry_pops(snr=10, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    #t = Table.read('../data/apogee_{:s}.fits'.format(tracer))
    #t = Table.read('../data/rcat_msto.fits')
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['eccen_pot1']<0.5) & (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.85)
    ind_sgr = t['Sgr_FLAG']==1
    #print(t.colnames)
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    ind_radial = (dlz>1) & (dlz<1.5)
    
    fehbins = np.linspace(-3,0.,40)
    
    plt.close()
    fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    #plt.plot(t['init_FeH'][ind_circular & ind_ridge], t['init_aFe'][ind_circular & ind_ridge], 'o', ms=3)
    plt.scatter(t['init_FeH'][ind_circular & ind_ridge], t['init_aFe'][ind_circular & ind_ridge], c=dlz[ind_circular & ind_ridge], s=5, vmin=0, vmax=0.8)
    
    plt.sca(ax[1])
    #plt.plot(t['init_FeH'][ind_circular & ind_ripple], t['init_aFe'][ind_circular & ind_ripple], 'o', ms=3)
    #plt.scatter(t['init_FeH'][ind_circular & ind_ripple], t['init_aFe'][ind_circular & ind_ripple], c=dlz[ind_circular & ind_ripple], s=5, vmin=0, vmax=0.8)
    
    #plt.plot(t['init_FeH'][ind_gse], t['init_aFe'][ind_gse], 'ro', ms=2, mew=0, zorder=0)
    #plt.plot(t['init_FeH'][ind_sgr], t['init_aFe'][ind_sgr], 'o', color='salmon', ms=2, mew=0, zorder=0)
    plt.scatter(t['init_FeH'][ind_circular], t['init_aFe'][ind_circular], c=dlz[ind_circular], s=5, vmin=0, vmax=1., zorder=1)
    
    #for ind in [ind_ridge, ind_ripple, ind_radial]:
        #plt.sca(ax[1])
        #plt.plot(t['init_FeH'][ind & ind_circular], t['init_aFe'][ind & ind_circular], 'o', ms=3)
        
        #plt.sca(ax[0])
        #plt.hist(t['init_FeH'][ind & ind_circular & (t['init_aFe']>0.25)], bins=fehbins, histtype='step', density=True)
    
    plt.xlim(-3,0.5)
    plt.tight_layout()



def rapo_rperi(snr=10, tracer='giants'):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['eccen_pot1']<0.5) & (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.85)
    ind_sgr = t['Sgr_FLAG']==1
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    for e, ind in enumerate([ind_circular, ind_sgr, ind_gse]):
        plt.sca(ax[e])
        plt.plot(t['Rperi_pot1'][ind], t['Rapo_pot1'][ind], 'k.', ms=1)
    
    plt.tight_layout()

def omega_comparison(snr=10, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['eccen_pot1']<0.5) & (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.85)
    ind_sgr = t['Sgr_FLAG']==1
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    
    for e, ind in enumerate([ind_circular, ind_sgr, ind_gse]):
        plt.plot(np.abs(t['omega_R'][ind].to(u.Gyr**-1)), np.abs(t['omega_phi'][ind].to(u.Gyr**-1)), '.', ms=1)
    
    plt.tight_layout()

def omegar_histogram(snr=10, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    #print(t.colnames)
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['eccen_pot1']<0.5) & (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.85)
    ind_sgr = t['Sgr_FLAG']==1
    
    elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115])*u.kpc**2*u.Myr**-2
    Nlevel = int(np.size(elevels)/2)
    de = np.abs(elevels[::2] - elevels[1::2])
    
    ind_energy = [(t['E_tot_pot1']>elevels[2*x]) & (t['E_tot_pot1']<elevels[2*x+1]) for x in range(Nlevel)]
    cmaps = [mpl.cm.Reds, mpl.cm.Oranges, mpl.cm.Greens, mpl.cm.Blues, mpl.cm.Purples]
    
    bins = np.linspace(2,80,150)
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)
    
    for e, ind in enumerate([ind_circular, ind_sgr, ind_gse]):
        plt.sca(ax[e])
        plt.hist(np.abs(t['omega_R'][ind].to(u.Gyr**-1).value), bins=bins, histtype='step')
        #plt.hist(2*np.pi*((t['orbit_period_pot1'][ind]*u.Myr).to(u.Gyr).value)**-1, bins=bins, histtype='step')
        
        #for i, inde in enumerate(ind_energy):
            #plt.hist(np.abs(t['omega_R'][ind & inde].to(u.Gyr**-1).value), bins=bins, histtype='step', color=cmaps[i](0.5))
    
    plt.tight_layout(h_pad=0)

def fft_omegar(snr=10, tracer='giants'):
    """"""
    
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr)
    t = t[ind]
    
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) #& (t['eccen_pot1']<0.5) & (t['Sgr_FLAG']==0)
    ind_gse = (t['eccen_pot1']>0.7)
    ind_sgr = t['Sgr_FLAG']==1
    
    labels = ['Disk', 'Sgr', 'GSE']
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    ind_radial = (dlz>1) & (dlz<1.5)
    
    omega_bar = (43*u.km/u.s/u.kpc).to(u.Gyr**-1)
    
    N = 2000
    bins = np.linspace(0,200,N)
    T = bins[1] - bins[0]
    
    plt.close()
    #fig, ax = plt.subplots(3,1,figsize=(10,8), sharex=True)
    fig, ax = plt.subplots(1,1,figsize=(10,6), sharex=True)
    
    for e, ind in enumerate([ind_circular, ind_sgr, ind_gse]):
        #plt.sca(ax[e])
        
        y, he = np.histogram(np.abs(t['omega_z'][ind].to(u.Gyr**-1).value), bins=bins)
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label=labels[e])
    
    for i in range(1,11):
        plt.axvline(i*omega_bar.value**-1, color='k', ls=':', lw=0.5)
    
    #print(2*np.pi*0.4)
    #print(2*np.pi*1.5)
    
    plt.legend()
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
        
    plt.tight_layout(h_pad=0)


def dlz_z(tracer='giants', snr=10):
    """"""
    
    plt.close()
    plt.figure()
    
    for tracer in ['msto', 'giants']:
        t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
        ind = (t['SNR']>snr)
        ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0)
        t = t[ind & ind_circular]
        
        par = np.load('../data/elz_ridgeline.npy')
        poly = np.poly1d(par)
        dlz = t['Lz'] - poly(t['E_tot_pot1'])
        
        plt.plot(dlz, np.abs(t['Z_gal']), 'o', mew=0, ms=4, alpha=0.5)
    
    plt.tight_layout()

def dlz_etot(snr=15):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>10)
    ind_circular = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['zmax_pot1']>2)
    t = t[ind & ind_circular]
    
    par = np.load('../data/elz_ridgeline.npy')
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    tmsto = Table.read('../data/rcat_msto.fits')
    ind = (tmsto['SNR']>20)
    ind_circular = (tmsto['circLz_pot1']>0.3) & (tmsto['Lz']<0) & (tmsto['zmax_pot1']>2)
    tmsto = tmsto[ind & ind_circular]
    
    print(np.median(t['E_tot_pot1_err']), np.median(tmsto['E_tot_pot1_err']))
    print(np.median(np.abs(t['Z_gal'])), np.median(np.abs(tmsto['Z_gal'])))
    print(np.nanmedian(t['zmax_pot1'][np.isfinite(t['zmax_pot1'])]))
    print(np.nanmedian(tmsto['zmax_pot1'][np.isfinite(tmsto['zmax_pot1'])]))
    
    dlz_msto = tmsto['Lz'] - poly(tmsto['E_tot_pot1'])
    
    ebins = np.linspace(-0.17,-0.08,70)
    min_giants = 0.15
    min_msto = 0.05
    
    ind_ripple = (dlz>min_giants) & (dlz<1)
    ind_ripple_msto = (dlz_msto>min_msto) & (dlz_msto<1)
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(9,9), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(t['E_tot_pot1'][ind_ripple], bins=ebins, histtype='step', color='b', density=True)
    plt.hist(tmsto['E_tot_pot1'][ind_ripple_msto], bins=ebins, histtype='step', color='r', density=True)
    
    plt.sca(ax[1])
    plt.plot(t['E_tot_pot1'], dlz, 'bo', mew=0, ms=2, alpha=0.4)
    plt.ylabel('$\Delta$ $L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.axhline(min_giants)
    plt.ylim(-0.2,1.2)
    
    plt.sca(ax[2])
    plt.plot(tmsto['E_tot_pot1'], dlz_msto, 'ro', mew=0, ms=2, alpha=0.4)
    plt.axhline(min_msto)
    
    #elevels = np.array([-0.15, -0.14, -0.136, -0.131, -0.13, -0.126, -0.125, -0.12, -0.119, -0.115, -0.111, -0.106])*u.kpc**2*u.Myr**-2
    #Nlevel = int(np.size(elevels)/2)
    #for j in range(Nlevel):
        #plt.axhspan(elevels[2*j].value, elevels[2*j+1].value, color=mpl.cm.Oranges_r(j/(Nlevel+1)), alpha=0.1)
    
    plt.ylim(-0.2,1.2)
    plt.xlim(-0.17, -0.08)
    
    plt.ylabel('$\Delta$ $L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout(h_pad=0)


def afeh_feathers():
    """"""
    
    ebands = np.array([[-0.152, -0.142], [-0.1344, -0.1307], [-0.128, -0.124], [-0.1210, -0.1166], [-0.1139, -0.1109], [-0.0976, -0.0921],])
    Nb = np.shape(ebands)[0]
    print(Nb)
    
    tall = Table.read('../data/rcat_giants.fits')
    ind = (tall['Lz']<0) & (tall['circLz_pot1']>0.4)
    t = tall[ind]
    
    plt.close()
    fig, ax = plt.subplots(Nb, 1, figsize=(12,12), sharex=True, sharey=True)
    
    for i in range(Nb):
        plt.sca(ax[Nb-1-i])
        ind = (t['E_tot_pot1']>ebands[i][0]) & (t['E_tot_pot1']<ebands[i][1])
        isort = np.argsort(t['circLz_pot1'][ind])[::-1]
        plt.plot(tall['FeH'], tall['aFe'], 'ko', ms=2, mew=0, alpha=0.1, zorder=0)
        plt.scatter(t['FeH'][ind][isort], t['aFe'][ind][isort], c=t['circLz_pot1'][ind][isort], vmin=0.4, vmax=0.9, s=5, zorder=1)
        
        plt.text(0.02,0.8, '{:.03f}<E$_{{tot}}$<{:.03f}'.format(ebands[i][0], ebands[i][1]), transform=plt.gca().transAxes, fontsize='small')
        
    plt.tight_layout(h_pad=0)
    
def afeh_circ():
    """"""
    
    tall = Table.read('../data/rcat_giants.fits')
    ind = (tall['Lz']<0) & (tall['circLz_pot1']>0.6) & (tall['E_tot_pot1']>-0.128*u.kpc**2*u.Myr**-2)
    t = tall[ind]
    
    ind_disk = (t['circLz_pot1']>0.77)
    ind_feather = (t['circLz_pot1']<0.77) & (t['circLz_pot1']>0.5)
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.plot(tall['FeH'], tall['aFe'], 'ko', ms=2, mew=0, alpha=0.4, zorder=0)
    plt.plot(t['FeH'][ind_disk], t['aFe'][ind_disk], 'ro', ms=2)
    plt.plot(t['FeH'][ind_feather], t['aFe'][ind_feather], 'bo', ms=2)
    
    plt.tight_layout()

def elz_afeh():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    
    p_higha = [-0.14,0.18]
    poly_higha = np.poly1d(p_higha)
    ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
    
    p_lowa = [-0.14,0.15]
    poly_lowa = np.poly1d(p_lowa)
    ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
    
    p_gse = [-0.32,-0.02]
    poly_gse = np.poly1d(p_gse)
    ind_gse = (t['init_FeH']<-0.6) & (t['init_aFe']<poly_gse(t['init_FeH'])) & (t['eccen_pot1']>0.7)
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(18,6), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    #plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    plt.plot(t['Lz'][ind_higha], t['E_tot_pot1'][ind_higha], 'ko', ms=2, mew=0)
    #plt.scatter(t['Lz'][ind_higha], t['E_tot_pot1'][ind_higha], c=t['FeH'][ind_higha], cmap='magma', s=3, vmin=-0.75, vmax=0)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.title('High [Fe/H], High [$\\alpha$/Fe]', fontsize='medium')
    
    plt.sca(ax[1])
    #plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    plt.plot(t['Lz'][ind_lowa], t['E_tot_pot1'][ind_lowa], 'ko', ms=2, mew=0)
    #plt.scatter(t['Lz'][ind_lowa], t['E_tot_pot1'][ind_lowa], c=t['FeH'][ind_lowa], cmap='magma', s=3, vmin=-0.75, vmax=0)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.title('High [Fe/H], Low [$\\alpha$/Fe]', fontsize='medium')
    
    plt.sca(ax[2])
    #plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
    plt.plot(t['Lz'][ind_gse], t['E_tot_pot1'][ind_gse], 'ko', ms=2, mew=0, alpha=0.7)
    #plt.scatter(t['Lz'][ind_lowa], t['E_tot_pot1'][ind_lowa], c=t['FeH'][ind_lowa], cmap='magma', s=3, vmin=-0.75, vmax=0)
    
    plt.xlim(-4, 4)
    plt.ylim(-0.18, -0.075)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.title('Low [Fe/H]', fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_afeh.png')
    
def rcirc(verbose=False, ret=False):
    """"""
    N = 1000
    pos = np.zeros((3,N))*u.kpc
    pos[1] = np.linspace(1,100,N)*u.kpc
    
    mw = ham.potential

    vcirc = mw.circular_velocity(pos)
    vel = np.ones((3,N))*u.km/u.s
    vel[0] = vcirc
    
    etot = mw.total_energy(pos, vel)
    
    #mw_z2 = gp.MilkyWayPotential(halo={'m': 0.8*5.4e11*u.Msun}, disk={'m':0.1*6.8e10*u.Msun})
    #vcirc_z2 = mw_z2.circular_velocity(pos)
    #vel_z2 = np.ones((3,N))*u.km/u.s
    #vel_z2[0] = vcirc_z2
    #etot_z2 = mw_z2.total_energy(pos, vel_z2)
    
    #eridge = np.array([-0.1475, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962, -0.059, -0.09975])
    eridge = np.array([-0.1475, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    
    # gaps
    eridge = np.array([-0.13869, -0.13054, -0.12373, -0.11603, -0.09975, -0.088, -0.059])

    #egap = np.array([-0.1475, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    
    ## second at solar circle
    #eridge = np.array([-0.1475, -0.1355, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    
    ## bottom of the lowest ridge -- check if similar to where GSE peters out! -> 6kpc, so pretty good agreement
    #eridge = np.array([-0.152, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    
    
    ind_close = [np.argmin(np.abs(etot - e*u.kpc**2*u.Myr**-2)) for e in eridge]
    r_ridge = np.array([pos[1][i].value for i in ind_close]) * pos[1][0].unit
    v_ridge = np.array([vel[0][i].value for i in ind_close]) * vel[0][0].unit
    t_ridge = 2*np.pi*(r_ridge/v_ridge).to(u.Gyr)
    
    #ind_close_z2 = [np.argmin(np.abs(etot_z2 - e*u.kpc**2*u.Myr**-2)) for e in eridge]
    #r_ridge_z2 = np.array([pos[1][i].value for i in ind_close_z2])
    
    if verbose:
        print(r_ridge)
        print(t_ridge)
        #print(t_ridge/t_ridge[-1])
        
        tr = t_ridge/t_ridge[-1]
        for tr_ in tr:
            fr_ = Fraction('{:f}'.format(tr_)).limit_denominator(10)
            print(fr_, np.abs(1 - fr_.numerator/fr_.denominator/tr_))
        #print(r_ridge_z2)
    
    if ret:
        return r_ridge

def rapo(selection='all', tracer='giants'):
    """"""
    
    if tracer=='giants':
        bins = np.arange(5,50,0.4)
        snr = 3
    elif tracer=='msto':
        bins = np.arange(5,50,0.1)
        snr = 10
    else:
        tracer = 'all'
        bins = np.arange(5,50,0.1)
        snr = 20
        
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr) & (t['Lz']<0)
    t = t[ind]
    
    if selection=='higha':
        p_higha = [-0.14,0.18]
        poly_higha = np.poly1d(p_higha)
        ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
        t = t[ind_higha]
        label = 'High [$\\alpha$/Fe]'
    elif selection=='lowa':
        p_lowa = [-0.14,0.15]
        poly_lowa = np.poly1d(p_lowa)
        ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
        t = t[ind_lowa]
        label = 'Low [$\\alpha$/Fe]'
        #bins = np.arange(5,50,0.4)
    else:
        selection = 'all'
        label = 'All'
    
    climit = 0.3
    ind_ = t['circLz_pot1']>climit
    
    ## R_circ from energy peaks
    #rc = rcirc()
    #N = np.size(rc)
    #cridge = [mpl.cm.magma(x/N) for x in range(N)]
    
    ## R_guide peaks
    #rg = np.array([])
    
    # gaps
    egap = np.array([-0.13869, -0.13054, -0.12373, -0.11603, -0.09975, -0.088, -0.059])
    N = 1000
    pos = np.zeros((3,N))*u.kpc
    pos[1] = np.linspace(1,100,N)*u.kpc
    
    mw = ham.potential
    vcirc = mw.circular_velocity(pos)
    vel = np.ones((3,N))*u.km/u.s
    vel[0] = vcirc
    etot = mw.total_energy(pos, vel)
    
    ind_close = [np.argmin(np.abs(etot - e*u.kpc**2*u.Myr**-2)) for e in egap]
    rgap = np.array([pos[1][i].value for i in ind_close]) * pos[1][0].unit
    vgap = np.array([vel[0][i].value for i in ind_close]) * vel[0][0].unit
    
    rgap = np.array([7.5539, 8.8218, 10.1037, 11.8885, 17.0875, 46])*u.kpc

    x_sgr = np.array([17.5,2.5,-6.5])*u.kpc
    dsgr = np.linalg.norm(x_sgr)
    
    c_lmc = coord.ICRS(ra=78.76*u.deg, dec=-69.19*u.deg, distance=10**(0.2*18.50+1)*u.pc,
                   radial_velocity=262.2*u.km/u.s,
                   pm_ra_cosdec=1.91*u.mas/u.yr, pm_dec=0.229*u.mas/u.yr)
    x_lmc = c_lmc.transform_to(coord.Galactocentric)
    dlmc = c_lmc.spherical.distance.to(u.kpc)
    print(dlmc)
    
    rgap = np.array([7.5539, 8.8218, 10.1037, 11.8885, 14.079, dsgr.value, dlmc.value])*u.kpc
    #rgap = np.array([7.572, 8.78013, 10.181, 11.9527, dsgr.value, dlmc.value])*u.kpc
    N = np.size(rgap)
    pos = np.zeros((3,N))*u.kpc
    pos[1] = rgap
    vgap = mw.circular_velocity(pos)
    
    tgap = 2*np.pi*(rgap/vgap).to(u.Gyr)
    
    
    tr = tgap / tgap[-1]
    fr = []
    for tr_ in tr:
        fr_ = Fraction('{:f}'.format(tr_)).limit_denominator(10)
        fr += [fr_]
        print(fr_, np.abs(1 - fr_.numerator/fr_.denominator/tr_))

    N = np.size(rgap)
    cgap = [mpl.cm.magma(x/N) for x in range(N)]
    
    
    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.hist(0.5*(t['Rperi_pot1'][ind_] + t['Rapo_pot1'][ind_]), bins=bins, histtype='step', lw=2, color='k', density=True)
    
    for i in range(N):
        plt.axvline(rgap[i].value, color=cgap[i])
    
    # Annotations
    labels = ['T$_{{circ}}$:$T_{LMC}$', 'T:$T_{LMC}$', 'T:$T_{LMC}$', 'T:$T_{LMC}$', '', 'Sgr', 'LMC']
    
    ymin, ymax = plt.gca().get_ylim()
    
    for i in range(N-1):
        if i<N-2:
            l = 'T$_{{circ}}$:$T_{{LMC}}$ = {:d}:{:d}'.format(fr[i].denominator, fr[i].numerator)
        else:
            l = labels[i]
        
        plt.text(0.99*rgap[i].value, 0.97*ymax, l, rotation=90, ha='right', va='top', fontsize='x-small')
    
    plt.text(0.02, 0.9, '{:s}'.format(label), transform=plt.gca().transAxes, ha='left')
    plt.xlabel('R$_{guide}$ [kpc]')
    plt.ylabel('Density')
    plt.xlim(5,20)
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('../plots/rguide_{:s}_{:s}.png'.format(tracer,selection))


def zvz(tracer='giants', selection='all'):
    """"""
    
    if tracer=='giants':
        bins = np.arange(5,50,0.2)
        snr = 3
    elif tracer=='msto':
        bins = np.arange(5,50,0.1)
        snr = 10
    else:
        tracer = 'all'
        bins = np.arange(5,50,0.1)
        snr = 20
        
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>snr) & (t['Lz']<0)
    t = t[ind]
    
    if selection=='higha':
        p_higha = [-0.14,0.18]
        poly_higha = np.poly1d(p_higha)
        ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
        t = t[ind_higha]
        label = 'High [$\\alpha$/Fe]'
    elif selection=='lowa':
        p_lowa = [-0.14,0.15]
        poly_lowa = np.poly1d(p_lowa)
        ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
        t = t[ind_lowa]
        label = 'Low [$\\alpha$/Fe]'
    else:
        selection = 'all'
        label = 'All'
    
    climit = 0.3
    ind_ = t['circLz_pot1']>climit
    
    plt.close()
    plt.figure()
    
    plt.plot(t['Z_gal'][ind_], t['Vz_gal'][ind_], 'ko', alpha=0.5, mew=0, ms=2)
    
    plt.xlim(-2,2)
    plt.ylim(-80,80)
    
    plt.tight_layout()


###########
# Disk spur

def elz_spur():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>10
    t = t[ind]
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    c = ['tab:red', 'tab:blue', 'darkorange']
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
        
        plt.xlim(-6,6)
        plt.ylim(-0.18, -0.02)
        
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    for i in range(3):
        plt.plot(t['Lz'][ind[i]], t['E_tot_pot1'][ind[i]], 'o', color=c[i], ms=2, mew=0)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_spur.png')

def elz_feathers():
    """"""
    N = 1000
    pos = np.zeros((3,N))*u.kpc
    pos[1] = np.linspace(1,100,N)*u.kpc
    
    mw = gp.MilkyWayPotential(halo={'m': 0.8*5.4e11*u.Msun}, disk={'m':0.1*6.8e10*u.Msun})
    mw = ham.potential

    vcirc = mw.circular_velocity(pos)
    vel = np.ones((3,N))*u.km/u.s
    vel[0] = vcirc
    
    etot = mw.total_energy(pos, vel)
    
    t = Table.read('../data/rcat_giants.fits')

    eridge = np.array([-0.1475, -0.1324, -0.1262, -0.1194, -0.1120, -0.0964])
    N = np.size(eridge)
    cridge = [mpl.cm.magma(x/N) for x in range(N)]
    
    ind_close = [np.argmin(np.abs(etot - e*u.kpc**2*u.Myr**-2)) for e in eridge]
    r_ridge = np.array([pos[1][i].value for i in ind_close])
    
    emin = -0.2

    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(11.5,6), sharey=True)
    
    plt.sca(ax[0])
    
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=1.5, mew=0, alpha=0.3, zorder=1)
    
    for k, e in enumerate(eridge):
        plt.axhline(e, color=cridge[k], lw=0.5, zorder=0, alpha=0.5)
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.plot(pos[1], etot, 'k-')
    
    plt.plot(r_ridge, eridge, 'ko', ms=4)
    
    for k, e in enumerate(eridge):
        r = r_ridge[k]
        x = np.array([0, r, r])
        y = np.array([e, e, emin])
        plt.plot(x, y, '-', color=cridge[k], lw=0.5, alpha=0.5)
        
        plt.text(r+0.2, e-0.005, '{:.1f} kpc'.format(r), fontsize='x-small')

    plt.xlim(0,40)
    plt.xlabel('R$_{circular}$ [kpc]')
    
    
    plt.tight_layout()
    plt.savefig('../plots/elz_feathers_rcirc.png')

def afeh_spur():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>10
    t = t[ind]
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    c = ['tab:red', 'tab:blue', 'darkorange']
    
    bins = np.linspace(-3,0.2,50)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,6), sharex=True)
    
    plt.sca(ax[0])
    plt.hist(t['FeH'], bins=bins, color='k', alpha=0.3, density=True)
    
    for i in range(3):
        plt.hist(t['FeH'][ind[i]], bins=bins, color=c[i], histtype='step', lw=2, density=True)
    
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    plt.plot(t['FeH'], t['aFe'], 'ko', ms=2, mew=0, alpha=0.3)
    
    for i in range(3):
        plt.plot(t['FeH'][ind[i]], t['aFe'][ind[i]], 'o', color=c[i], ms=5, mew=0)
    
    plt.gca().set_aspect('equal', adjustable='datalim')
    
    plt.xlabel('[Fe/H]')
    plt.ylabel('[$\\alpha$/Fe]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/afeh_spur.png')

def sky_spur():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>10
    t = t[ind]
    print(len(t))
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    color = ['tab:red', 'tab:blue', 'darkorange']
    
    c = coord.SkyCoord(ra=t['RA']*u.deg, dec=t['DEC']*u.deg, distance=t['dist_adpt']*u.kpc, pm_ra_cosdec=t['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=t['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=t['Vrad']*u.km/u.s, frame='icrs')
    cgal = c.transform_to('galactic')
    wangle = 180*u.deg
    l0 = 120*u.deg
    
    print(np.sum(ind_off), np.sum(ind_disk))
    
    plt.close()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(projection='hammer')
    
    plt.plot((cgal.l - l0).wrap_at(wangle).radian, cgal.b.radian, 'ko', ms=2, mew=0, alpha=0.3)
    
    for i in range(3):
        plt.plot((cgal.l - l0).wrap_at(wangle).radian[ind[i]], cgal.b.radian[ind[i]], 'o', c=color[i])
    
    plt.xlabel('l - $l_0$ [deg]')
    plt.ylabel('b [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/sky_spur.png')

    
    plt.tight_layout()

def lperp_spur():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>10
    t = t[ind]
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    c = ['tab:red', 'tab:blue', 'darkorange']
    
    bins = np.linspace(0,2,30)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,5), sharey=True)
    
    plt.sca(ax[0])
    plt.plot(t['Lz'], t['Lperp'], 'ko', ms=2, mew=0, alpha=0.3)
    
    for i in range(2):
        plt.plot(t['Lz'][ind[i]], t['Lperp'][ind[i]], 'o', color=c[i], ms=6, mew=0)
    
    plt.xlim(-4,0)
    plt.ylim(0,2)
    #plt.gca().set_aspect('equal')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$L_\perp$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.sca(ax[1])
    plt.hist(t['Lperp'], bins=bins, color='k', alpha=0.3, orientation='horizontal', density=True)
    
    for i in range(2):
        plt.hist(t['Lperp'][ind[i]], bins=bins, color=c[i], lw=2, histtype='step', orientation='horizontal', density=True)
        
    plt.tight_layout()
    plt.savefig('../plots/lzlperp_spur.png')

def l_spur():
    """"""
    
    t = Table.read('../data/rcat_giants.fits')
    ind = t['SNR']>10
    t = t[ind]
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    c = ['tab:red', 'tab:blue', 'darkorange']
    
    plt.close()
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    
    plt.sca(ax[0])
    for i in range(2):
        plt.plot(t['Lz'][ind[i]], t['Lperp'][ind[i]], 'o', color=c[i], ms=6, mew=0)
    
    plt.sca(ax[1])
    for i in range(2):
        plt.plot(t['Lx'][ind[i]], t['Ly'][ind[i]], 'o', color=c[i], ms=6, mew=0)
    
    plt.sca(ax[2])
    for i in range(2):
        plt.plot(t['Lz'][ind[i]], t['Ltot'][ind[i]], 'o', color=c[i], ms=6, mew=0)
    
    
    plt.tight_layout()

def circularity_spur(tracer='giants'):
    """"""
    t = Table.read('../data/rcat_{:s}.fits'.format(tracer))
    ind = (t['SNR']>10) & (t['Lz']<0)
    t = t[ind]
    
    #p_higha = [-0.14,0.18]
    #poly_higha = np.poly1d(p_higha)
    #ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
    #t = t[ind_higha]

    
    eridge = np.array([-0.1475, -0.1324, -0.1262, -0.1194, -0.1120, -0.0964])
    eridge = np.array([-0.1485, -0.1455, -0.1326, -0.1269, -0.1228, -0.1196, -0.114, -0.1094, -0.105, -0.1005, -0.0938])

    N = np.size(eridge)
    cridge = [mpl.cm.magma(x/N) for x in range(N)]
    climit = 0.4
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    plt.sca(ax[0])
    ind_ = t['circLz_pot1']>climit
    bins = np.linspace(-0.17,-0.06, 300)
    
    plt.hist(t['E_tot_pot1'][ind_], bins=bins, density=True, color='k', lw=2, histtype='step')
    
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    plt.plot(t['E_tot_pot1'], t['circLz_pot1'], 'ko', ms=2, mew=0, alpha=0.4)
    
    plt.axhline(climit, color='r')
    plt.xlim(-0.18,-0.1)

    for i in range(2):
        plt.sca(ax[i])
        for k, e in enumerate(eridge):
            plt.axvline(e, color=cridge[k], lw=0.5, zorder=0, alpha=0.5)
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Mpc$^{-2}$]')
    plt.ylabel('Circularity')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/circularity_feathers_all_{:s}.png'.format(tracer))

def circ_selection(selection='all'):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0.) #& (t['eccen_pot1']>0.7)
    t = t[ind]
    
    if selection=='higha':
        p_higha = [-0.14,0.18]
        poly_higha = np.poly1d(p_higha)
        ind_higha = (t['init_FeH']>-0.75) & (t['init_aFe']>poly_higha(t['init_FeH']))
        t = t[ind_higha]
    elif selection=='lowa':
        p_lowa = [-0.14,0.15]
        poly_lowa = np.poly1d(p_lowa)
        ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
        t = t[ind_lowa]
    else:
        selection = 'all'
    
    eridge = np.array([-0.1475, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    eridge = np.array([-0.15106, -0.1455, -0.1326, -0.1269, -0.1228, -0.1196, -0.114, -0.1094, -0.105, -0.1005, -0.0938])
    #print(eridge[1:-1]-eridge[2:])
    N = np.size(eridge)
    cridge = [mpl.cm.magma(x/N) for x in range(N)]
    climit = 0.4
    
    ebands = np.array([[-0.152, -0.142], [-0.1344, -0.1307], [-0.128, -0.124], [-0.1210, -0.1166], [-0.1139, -0.1109], [-0.1074, -0.1048], [-0.0976, -0.0921],])
    N = np.shape(ebands)[0]
    #print(N)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,8), sharex=True)
    
    plt.sca(ax[0])
    ind_circ = t['circLz_pot1']>climit
    ind_rad_mpoor = (t['circLz_pot1']<climit) & (t['eccen_pot1']>0.7) #& (t['FeH']<-1)
    ind_rad = (t['circLz_pot1']<climit) & (t['eccen_pot1']>0.7)
    bins = np.linspace(-0.17,-0.06, 90)
    
    plt.hist(t['E_tot_pot1'][ind_circ], bins=bins, density=True, color='0.5', lw=2, histtype='step', label='circ>{:g}'.format(climit))
    plt.hist(t['E_tot_pot1'][ind_rad_mpoor], bins=bins, density=True, color='k', lw=2, histtype='step', label='circ<{:g} & ecc>0.7'.format(climit))
    #plt.hist(t['E_tot_pot1'][ind_rad], bins=bins, density=True, color='0.8', lw=2, histtype='step', label='circ<{:g} & ecc>0.7'.format(climit))
    
    plt.legend(fontsize='small')
    plt.ylabel('Density')
    
    plt.sca(ax[1])
    #plt.plot(t['E_tot_pot1'], t['circLz_pot1'], 'ko', ms=2, mew=0, alpha=0.4)
    plt.plot(t['E_tot_pot1'][ind_circ], t['circLz_pot1'][ind_circ], 'ko', ms=2, mew=0, alpha=0.4)
    plt.plot(t['E_tot_pot1'][ind_rad], t['circLz_pot1'][ind_rad], 'ko', ms=2, mew=0, alpha=0.4)
    
    plt.axhline(climit, color='r')
    plt.xlim(-0.18,-0.05)

    #for i in range(2):
        #plt.sca(ax[i])
        #for k, e in enumerate(eridge):
            #plt.axvline(e, color=cridge[k], lw=0.5, zorder=0, alpha=0.5)
        
        #for k in range(N):
            #plt.axvspan(ebands[k][0], ebands[k][1], color=cridge[k], alpha=0.2)
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Mpc$^{-2}$]')
    plt.ylabel('Circularity')
    plt.ylim(0,1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/circ_{:s}.png'.format(selection))

def msto_spur():
    """"""
    t = Table.read('../data/rcat_msto.fits')
    ind = t['SNR']>15
    t = t[ind]
    
    ind_spur = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_disk = (t['Lz']<-1.27*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1355*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.131*u.kpc**2*u.Myr**-2)
    ind_off = (t['Lz']>-1.27*u.kpc**2*u.Myr**-1) & (t['Lz']<-0.76*u.kpc**2*u.Myr**-1) & (t['E_tot_pot1']>-0.1395*u.kpc**2*u.Myr**-2) & (t['E_tot_pot1']<-0.1355*u.kpc**2*u.Myr**-2)
    
    ind = [ind_spur, ind_disk, ind_off]
    c = ['tab:red', 'tab:blue', 'darkorange']
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    for i in range(2):
        plt.sca(ax[i])
        plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.3)
        
        plt.xlim(-6,6)
        plt.ylim(-0.18, -0.02)
        
        plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
        plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    for i in range(3):
        plt.plot(t['Lz'][ind[i]], t['E_tot_pot1'][ind[i]], 'o', color=c[i], ms=2, mew=0)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_msto_spur.png')


def gse_disk():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.4) & (t['Lz']<0)
    ind_gse = (t['eccen_pot1']>0.7)
    
    t_disk = t[ind_disk]
    t_gse = t[ind_gse]
    
    ind_prograde = t_gse['Lz']<0.
    ind_retrograde = (t_gse['Lz']>0) #& (t_gse['Lz']<0.5)
    
    bins = np.linspace(-0.17,-0.06, 90)
    
    plt.close()
    plt.figure(figsize=(14,6))
    #fig, ax = plt.subplots(2,1,figsize=(12,8))
    
    plt.hist(t_gse['E_tot_pot1'][ind_prograde], bins=bins, histtype='stepfilled', density=True, color='b', alpha=0.5, label='GSE prograde')
    #plt.hist(t_gse['E_tot_pot1'][ind_retrograde], bins=bins, histtype='step', density=True, color='k', alpha=0.5, label='GSE retrograde')
    #plt.hist(t_gse['E_tot_pot1'], bins=bins, histtype='stepfilled', density=True, color='b', alpha=0.5, label='GSE')
    plt.hist(t_disk['E_tot_pot1'], bins=bins, histtype='stepfilled', density=True, color='r', alpha=0.5, label='Disk')
    
    plt.legend()

    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('Density')
    
    plt.tight_layout()

def period_hist_populations(offset=True, snr=5):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>snr)
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>snr) # & (t['FeH']<-1)
    ind_sgr = (t['Sgr_FLAG']==1) & (t['SNR']>snr)
    
    print(len(t))
    #t_disk = t[ind_disk]
    #t_gse = t[ind_gse]
    
    ## APOGEE
    #ta = Table(fits.getdata('../data/apogee_giants.fits'))
    #ind = (ta['Lz']<-0.5) & (ta['dist']>500)
    ##print(np.percentile(ta['dist'], [16,50,84]))
    #ta = ta[ind]
    #ind_high = (ta['zmax']>2)
    #ind_low = (ta['zmax']<0.3)
    #ta = ta[ind_low]
    ##ta = ta[ind_high]
    #Pa = ta['orbit_period_pot1'] / 947
    
    rguide = 0.5*(t['Rperi_pot1']+t['Rapo_pot1'])
    rbins = np.linspace(2,30,80)
    
    P = t['orbit_period_pot1'] / 947
    pbins = np.linspace(0.06,0.6,80)
    
    pres = np.array([1/2, 5/11., 2/5., 1/3., 3/10., 3/11., 1/4., 2/9., 1/5., 1/6., 1/7., 1/10.])
    pres = np.array([1/2, 1/3., 3/10., 3/11., 1/4., 2/9., 1/5., 1/6., 1/7., 1/10., 1/14.])
    Nres = np.size(pres)
    
    fres = []
    for pr in pres:
        fr = Fraction('{:f}'.format(pr)).limit_denominator(14)
        fres += [fr]
        #print(fr, np.abs(1 - fr.numerator/fr.denominator/pr))
    
    lw = 2
    
    if offset:
        off = np.array([0.005, 0.015, 0.003])
        #off = np.array([0.005, 0.01, 0.003])
    else:
        off = np.zeros(3)
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,9), sharex=True)
    
    #plt.hist(rguide[ind_gse], histtype='step', bins=rbins, density=True)
    #plt.hist(rguide[ind_disk], histtype='step', bins=rbins, density=True)
    
    plt.sca(ax[0])
    plt.hist(P[ind_gse]-off[0], histtype='step', bins=pbins, density=True, lw=lw, color='navy')
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'H3 GSE\n(ecc>0.7) & ($L_Z$<0)\nP/P$_{{Sgr}}$ - {:g}'.format(off[0]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    plt.sca(ax[1])
    plt.hist(P[ind_disk]-off[1], histtype='step', bins=pbins, density=True, lw=lw, color='r')
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'H3 Disk\n(circ>0.3) & ($L_Z$<0)\nP/P$_{{Sgr}}$ - {:g}'.format(off[1]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    #for rlim in [10,12,15,20]:
        #ind_far = t['R_gal']>rlim
        #plt.hist(P[ind_disk & ind_far]-0.015, histtype='step', bins=pbins, density=True)
    
    plt.sca(ax[2])
    #plt.hist(Pa-off[2], histtype='step', bins=pbins, density=True, lw=lw, color='orange')
    plt.hist(P[ind_sgr]-off[2], histtype='step', bins=pbins, density=True, lw=lw, color='orange')
    plt.ylabel('Density')
    plt.xlabel('Orbital period / Sgr orbital period')
    plt.text(0.98,0.95, 'APOGEE Thin disk\n($Z_{{max}}<0.3$kpc) & ($L_Z$<-0.5kpc$^2$Myr$^{{-1}}$) & ($d>0.5$kpc)\nP/P$_{{Sgr}}$ - {:g}'.format(off[2]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    # draw resonance lines
    for i in range(3):
        plt.sca(ax[i])
        for res in pres:
            plt.axvline(res, color='k', lw=0.5, alpha=0.5)
    
    # resonance labels
    plt.sca(ax[0])
    ymax = 5
    for i in range(Nres):
        l = 'P$_{{orb}}$:$P_{{Sgr,orb}}$ = {0:d}:{1:d}'.format(fres[i].denominator, fres[i].numerator)
        l = '{0:d}:{1:d}'.format(fres[i].denominator, fres[i].numerator)
        
        plt.text(0.99*pres[i], 0.97*ymax, l, rotation=90, ha='right', va='top', fontsize='x-small')
    
    plt.ylim(0,5)
    
    plt.tight_layout(h_pad=0)
    #plt.savefig('../plots/period_hist_populations_off.{:d}.png'.format(offset))

def periods():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>3)
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>3) & (t['omega_z']>0.001)# & (t['FeH']<-1)
    ind_sgr = (t['Sgr_FLAG']==1) & (t['omega_z']>0.001)
    
    Nbin = 100
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,8))
    
    for e, omega in enumerate(['omega_R', 'omega_phi', 'omega_z']):
        plt.sca(ax[e])
        plt.hist(np.abs(t[omega][ind_disk]), bins=Nbin, color='r', histtype='step')
        plt.hist(np.abs(t[omega][ind_gse]), bins=Nbin, color='b', histtype='step')
        plt.hist(np.abs(t[omega][ind_sgr]), bins=Nbin, color='g', histtype='step')
    
    plt.tight_layout()

def frequencies_2d():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>3)
    ind_gse = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>3) #& (t['omega_z']>0.001)# & (t['FeH']<-1)
    ind_sgr = t['Sgr_FLAG']==1
    
    plt.close()
    plt.figure()
    
    plt.plot(np.abs(t['omega_R'][ind_disk])*1e3/(2*np.pi), np.abs(t['omega_z'][ind_disk])*1e3/(2*np.pi), 'r.')
    plt.plot(np.abs(t['omega_R'][ind_gse])*1e3/(2*np.pi), np.abs(t['omega_z'][ind_gse])*1e3/(2*np.pi), 'b.')
    plt.plot(np.abs(t['omega_R'][ind_sgr])*1e3/(2*np.pi), np.abs(t['omega_z'][ind_sgr])*1e3/(2*np.pi), 'g.')
    
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    
    plt.tight_layout()

def etot_hist_populations(zmax=False):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_disk = (t['circLz_pot1']>0.3) & (t['Lz']<0) & (t['SNR']>5)
    ind_gse = (t['eccen_pot1']>0.7) & (t['SNR']>5) # & (t['Lz']<0.) #& (t['FeH']<-1)
    ind_gse_pro = (t['eccen_pot1']>0.7) & (t['Lz']<0.) & (t['SNR']>5) # & (t['FeH']<-1)
    
    p_lowa = [-0.14,0.15]
    poly_lowa = np.poly1d(p_lowa)
    ind_lowa = (t['init_FeH']>-0.45) & (t['init_aFe']<poly_lowa(t['init_FeH']))
    ind_zmax = (t['zmax_pot1']>6)
    
    # APOGEE
    ta = Table(fits.getdata('../data/apogee_giants.fits'))
    ind = (ta['Lz']<-0.5) & (ta['dist']>500)
    ta = ta[ind]
    ind_high = (ta['zmax']>2)
    ind_low = (ta['zmax']<0.3)
    ta = ta[ind_low]
    Pa = ta['orbit_period_pot1'] / 947
    
    rguide = 0.5*(t['Rperi_pot1']+t['Rapo_pot1'])
    rbins = np.linspace(2,30,80)
    
    P = t['orbit_period_pot1'] / 947
    pbins = np.linspace(0.06,0.6,80)
    
    pres = np.array([1/2, 5/11., 2/5., 1/3., 3/10., 3/11., 1/4., 2/9., 1/5., 1/6., 1/7., 1/10.])
    pres = np.array([1/2, 1/3., 3/10., 3/11., 1/4., 2/9., 1/5., 1/6., 1/7., 1/10., 1/14.])
    Nres = np.size(pres)
    
    fres = []
    for pr in pres:
        fr = Fraction('{:f}'.format(pr)).limit_denominator(14)
        fres += [fr]
        #print(fr, np.abs(1 - fr.numerator/fr.denominator/pr))
    
    lw = 2
    
    off = np.zeros(3)
    
    Esgr = -0.06341629#*u.kpc**2*u.Myr**-2
    E = t['E_tot_pot1']#/Esgr
    Ea = ta['E_tot_pot1']#/Esgr
    ebins = np.linspace(-0.16,-0.06,100)
    ebins = np.linspace(-0.16,-0.06,75)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(12,9), sharex=True)
    
    egap = [-0.1393, -0.13129, -0.12358, -0.11891, -0.11485, -0.10511, -0.09588]
    
    for e in egap:
        for i in range(2):
            plt.sca(ax[i])
            plt.axvline(e, color='k', ls=':', lw=0.5)
    
    plt.sca(ax[0])
    #plt.hist(E[ind_gse], histtype='step', bins=ebins, density=True, lw=lw, color='navy')
    plt.hist(E[ind_gse_pro], histtype='step', bins=ebins, density=True, lw=lw, color='tab:blue')
    plt.ylabel('Density')
    plt.text(0.98,0.95, 'H3 GSE\n(ecc>0.7) & ($L_Z$<0)'.format(off[0]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    plt.sca(ax[1])
    if zmax:
        plt.hist(E[ind_disk & ind_zmax]-off[1], histtype='step', bins=ebins, density=True, lw=lw, color='r')
    else:
        plt.hist(E[ind_disk]-off[1], histtype='step', bins=ebins, density=True, lw=lw, color='r')
    plt.ylabel('Density')
    if zmax:
        plt.text(0.98,0.95, 'H3 Disk\n(circ>0.3) & ($L_Z$<0) & ($Z_{{max}}$>6kpc)'.format(off[1]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    else:
        plt.text(0.98,0.95, 'H3 Disk\n(circ>0.3) & ($L_Z$<0)'.format(off[1]), va='top', ha='right', transform=plt.gca().transAxes, fontsize='small')
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/etot_hist_populations_zmax.{:d}.png'.format(zmax))


def epeaks_rvr(gse=False):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind_gse = (t['eccen_pot1']>0.7) & (t['SNR']>5) # & (t['FeH']<-1)
    #tg = t[ind_gse]
    ind_e1 = (t['E_tot_pot1']>-0.1375) & (t['E_tot_pot1']<-0.1325)
    ind_e2 = (t['E_tot_pot1']>-0.1275) & (t['E_tot_pot1']<-0.1245)
    ind_e3 = (t['E_tot_pot1']>-0.1125) & (t['E_tot_pot1']<-0.1075)
    
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', ms=2, mew=0, alpha=0.2)
    plt.xlim(-6,6)
    plt.ylim(-0.18,-0.02)
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.plot(t['R_gal'], t['Vr_gal'], 'ko', ms=2, mew=0, alpha=0.2)
    
    #plt.gca().set_xscale('log')
    plt.xlim(5,50)
    plt.ylim(-400,400)
    plt.xlabel('$R_{gal}$ [kpc]')
    plt.ylabel('$V_{r,gal}$ [km s$^{-1}$]')
    
    if gse:
        for ind_e in [ind_e1, ind_e2, ind_e3]:
            plt.sca(ax[0])
            plt.plot(t['Lz'][ind_gse & ind_e], t['E_tot_pot1'][ind_gse & ind_e], 'o', ms=2, mew=0)
            
            plt.sca(ax[1])
            plt.plot(t['R_gal'][ind_gse & ind_e], t['Vr_gal'][ind_gse & ind_e], 'o', ms=2, mew=0)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_rvr_gse.{:d}.png'.format(gse))


def elz_edge(component='disk'):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    
    if component=='disk':
        ind_comp = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']<0.7) #(t['circLz_pot1']>0.3)
        ymin = -6
        sigma_smooth = 0.2
    else:
        ind_comp = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']>0.7)
        component = 'gse'
        ymin = -2
        sigma_smooth = 0.25
    
    t = t[ind_comp]
    
    print(np.median(t['Lz_err']))
    print(np.median(t['E_tot_pot1_err']))
    
    
    # calculate normalized Lz
    if component=='disk':
        # envelope
        Nbin = 20
        ebins = np.linspace(-0.17,-0.08,Nbin)
        de = 0.01
    
        lz_max = np.empty(Nbin)
        pmax = 5
        
        for i in range(Nbin):
            ind_ = (t['E_tot_pot1']>ebins[i]-de) & (t['E_tot_pot1']<ebins[i]+de)
            lz_max[i] = np.percentile(t['Lz'][ind_], pmax)
        
        kfit = 2
        pfit = np.polyfit(ebins, lz_max, kfit)
        poly = np.poly1d(pfit)
        
        lz_norm = t['Lz'] - poly(t['E_tot_pot1']) - 0.2
        
    else:
        lz_norm = np.abs(t['Lz'])

    ebins = np.arange(-0.155,-0.098,0.001)
    Nbin = np.size(ebins)
    de = 0.001
    
    # Lz track
    lz_track = np.empty(Nbin)
    
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(14,10), sharex='col')
    
    plt.sca(ax[0][0])
    plt.plot(t['E_tot_pot1'], t['Lz'], 'ko', mew=0, ms=2)
    
    plt.ylim(0, ymin)
    plt.xlim(-0.18,-0.08)
    plt.ylabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1][0])
    plt.plot(t['E_tot_pot1'], lz_norm, 'ko', mew=0, ms=4)
    plt.ylim(0,2)
    plt.xlim(-0.18,-0.08)
    plt.ylabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[0][1])
    plt.xlabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('KDE Density')
    
    plt.sca(ax[1][1])
    plt.xlabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('KDE Derivative')
    
    for i in range(Nbin):
        ind_ = (t['E_tot_pot1']>ebins[i]-de) & (t['E_tot_pot1']<ebins[i]+de)
        lz_max = min(1.5, np.max(lz_norm[ind_]))
        
        # make sure energy bins are not smaller than typical energy uncertainty
        de_ = max(de, 0.5 * np.median(t['E_tot_pot1_err'][ind_]))
        ind_ = (t['E_tot_pot1']>ebins[i]-de_) & (t['E_tot_pot1']<ebins[i]+de_) & (lz_norm<lz_max)
        #print(de_)
        
        sigma_lz = np.sqrt(sigma_smooth**2 + (np.median(t['Lz_err'][ind_]))**2)
        kde = gaussian_kde(lz_norm[ind_], bw_method=sigma_lz)
        c = mpl.cm.viridis(i/Nbin)
        
        x = np.linspace(0,lz_max,10000)
        y = kde(x)
        
        grad = np.gradient(y)
        grad2 = np.gradient(grad)
        
        xtol = np.percentile(np.abs(grad2),[3])
        ind_root = (np.abs(grad2)<xtol) & (grad<0)
        ind_min = 0
        #ind_min = np.argmin(grad[ind_root])
        
        lz_track[i] = x[ind_root][ind_min]
        
        plt.sca(ax[0][1])
        plt.plot(x, y, '-', color=c, lw=0.5)
        
        plt.sca(ax[1][1])
        plt.plot(x, grad, '-', color=c, lw=0.5)
        plt.plot(x[ind_root][ind_min], grad[ind_root][ind_min], 'o', color=c, lw=0.5)
        
        plt.sca(ax[1][0])
        plt.plot(ebins[i], lz_track[i], 'o', color=c, ms=3)
        
    np.savez('../data/elz_edge_{:s}.npz'.format(component), etot=ebins, lznorm=lz_track)

    plt.tight_layout()
    plt.savefig('../plots/elz_edge_{:s}.png'.format(component))

def elz_edge_bootstrap(full=False, Nboot=100, component='disk'):
    """"""
    t = Table.read('../data/rcat_giants.fits')
    
    if component=='disk':
        ind_comp = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']<0.7) #(t['circLz_pot1']>0.3)
        ymin = -6
        sigma_smooth = 0.2
    else:
        ind_comp = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']>0.7)
        component = 'gse'
        ymin = -2
        sigma_smooth = 0.2
    
    t = t[ind_comp]
    N = len(t)
    
    seed = 381
    np.random.seed(seed)
    
    if full:
        #Nboot = 500 # do a resolution study by increasing bootstrap
        disk_off = 0.2
        Ndim = 6
        offsets = np.random.randn(N,Nboot,Ndim)
        
        ra = (t['GAIAEDR3_RA'][:,np.newaxis] + offsets[:,:,0] * t['GAIAEDR3_RA_ERROR'][:,np.newaxis]) * u.deg
        dec = (t['GAIAEDR3_DEC'][:,np.newaxis] + offsets[:,:,1] * t['GAIAEDR3_DEC_ERROR'][:,np.newaxis]) * u.deg
        dist = (t['dist_adpt'][:,np.newaxis] + offsets[:,:,2] * t['dist_adpt_err'][:,np.newaxis]) * u.kpc
        dist[dist<0*u.kpc] = 0*u.kpc
        pmra = (t['GAIAEDR3_PMRA'][:,np.newaxis] + offsets[:,:,3] * t['GAIAEDR3_PMRA_ERROR'][:,np.newaxis]) * u.mas/u.yr
        pmdec = (t['GAIAEDR3_PMDEC'][:,np.newaxis] + offsets[:,:,4] * t['GAIAEDR3_PMDEC_ERROR'][:,np.newaxis]) * u.mas/u.yr
        vr = (t['Vrad'][:,np.newaxis] + offsets[:,:,5] * t['Vrad_err'][:,np.newaxis]) * u.km/u.s
        
        c = coord.SkyCoord(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=0)
        
        etot = orbit.energy()[0]#.reshape(N,-1)
        lz = orbit.angular_momentum()[2][0]#.reshape(N,-1)
    else:
        disk_off = 0.35
        Ndim = 2
        offsets = np.random.randn(N,Nboot,Ndim)
        
        etot = (t['E_tot_pot1'][:,np.newaxis] + t['E_tot_pot1_err'][:,np.newaxis] * offsets[:,:,1]) * u.kpc**2*u.Myr**-2
        lz = (t['Lz'][:,np.newaxis] + t['Lz_err'][:,np.newaxis] * offsets[:,:,1]) * u.kpc**2*u.Myr**-1
    
    
    # calculate normalized Lz
    if component=='disk':
        # envelope
        Nbin = 20
        ebins = np.linspace(-0.17,-0.08,Nbin)
        de = 0.01
    
        lz_max = np.empty(Nbin)
        pmax = 5
        
        for i in range(Nbin):
            ind_ = (etot.value>ebins[i]-de) & (etot.value<ebins[i]+de)
            lz_max[i] = np.percentile(lz.value[ind_], pmax)
        
        kfit = 2
        pfit = np.polyfit(ebins, lz_max, kfit)
        poly = np.poly1d(pfit)
        
        lz_norm = lz.value - poly(etot.value) - disk_off
        
    else:
        lz_norm = np.abs(lz.value)

    ebins = np.arange(-0.155,-0.098,0.001)
    Nbin = np.size(ebins)
    de = 0.001
    
    # Lz track
    lz_track = np.empty(Nbin)
    
    plt.close()
    fig, ax = plt.subplots(2,2,figsize=(14,10), sharex='col')
    
    plt.sca(ax[0][0])
    plt.plot(etot, lz, 'ko', mew=0, ms=1, alpha=0.1)
    
    plt.ylim(0, ymin)
    plt.xlim(-0.18,-0.08)
    plt.ylabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1][0])
    plt.plot(etot, lz_norm, 'ko', mew=0, ms=1, alpha=0.1)
    plt.ylim(0,2)
    plt.xlim(-0.18,-0.08)
    plt.ylabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[0][1])
    plt.xlabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('KDE Density')
    
    plt.sca(ax[1][1])
    plt.xlabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('KDE Derivative')
    
    
    for i in range(Nbin):
        ind_ = (etot.value>ebins[i]-de) & (etot.value<ebins[i]+de)
        lz_max = min(1.5, np.max(lz_norm[ind_]))
        
        # make sure energy bins are not smaller than typical energy uncertainty
        de_ = max(de, 0.5 * np.median(etot.value[ind_]))
        ind_ = (etot.value>ebins[i]-de_) & (etot.value<ebins[i]+de_) & (lz_norm<lz_max)
        #print(de_)
        
        #sigma_lz = np.sqrt(sigma_smooth**2 + (np.median(t['Lz_err'][ind_]))**2)
        sigma_lz = sigma_smooth
        kde = gaussian_kde(lz_norm[ind_], bw_method=sigma_lz)
        c = mpl.cm.viridis(i/Nbin)
        
        x = np.linspace(0,lz_max,10000)
        y = kde(x)
        
        grad = np.gradient(y)
        grad2 = np.gradient(grad)
        
        xtol = np.percentile(np.abs(grad2),[3])
        ind_root = (np.abs(grad2)<xtol) & (grad<0)
        ind_min = 0
        ## this picks up separate clumps
        #ind_min = np.argmin(grad[ind_root])
        
        lz_track[i] = x[ind_root][ind_min]
        
        plt.sca(ax[0][1])
        plt.plot(x, y, '-', color=c, lw=0.5)
        
        plt.sca(ax[1][1])
        plt.plot(x, grad, '-', color=c, lw=0.5)
        plt.plot(x[ind_root][ind_min], grad[ind_root][ind_min], 'o', color=c, lw=0.5)
        
        plt.sca(ax[1][0])
        plt.plot(ebins[i], lz_track[i], 'o', color=c, ms=3)
        
    np.savez('../data/elz_edge_bootstrap_{:s}.npz'.format(component), etot=ebins, lznorm=lz_track)
    
    plt.tight_layout()
    plt.savefig('../plots/elz_edge_bootstrap_{:s}.png'.format(component))


def elz_edge_comparison(boot=False):
    """"""
    if boot:
        label = '_bootstrap'
    else:
        label = ''
    td = np.load('../data/elz_edge{:s}_disk.npz'.format(label))
    th = np.load('../data/elz_edge{:s}_gse.npz'.format(label))
    
    plt.close()
    
    plt.plot(td['etot'], td['lznorm']-0.15, 'r-', label='Disk')
    plt.plot(th['etot'], th['lznorm'], 'b-', label='GSE')
    
    #plt.gca().set_yscale('log')
    plt.legend(fontsize='small')
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('$\\tilde{L}_z$ [kpc$^2$ Myr$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/elz_edge{:s}_comparison.png'.format(label))

def etot_kde():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    
    ind_disk = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']<0.7) & (t['circLz_pot1']>0.3)
    ind_gse = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']>0.7)
    
    bw = 0.03
    smooth_disk = np.median(t['E_tot_pot1_err'][ind_disk]) + bw
    smooth_gse = np.median(t['E_tot_pot1_err'][ind_gse]) + bw
    
    kde_disk = gaussian_kde(t['E_tot_pot1'][ind_disk], bw_method=smooth_disk)
    kde_gse = gaussian_kde(t['E_tot_pot1'][ind_gse], bw_method=smooth_gse)
    
    x = np.linspace(-0.18,-0.09,1000)
    etot_disk = kde_disk(x)
    etot_gse = kde_gse(x)
    
    plt.close()
    plt.figure()
    
    plt.plot(x, etot_disk, 'r-')
    plt.plot(x, etot_gse, 'b-')
    
    plt.tight_layout()


# moving group spaces

def lz_phi():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    
    ind_disk = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']<0.7) & (t['circLz_pot1']>0.3)
    ind_gse = (t['Lz']<0) & (t['SNR']>3) & (t['eccen_pot1']>0.7)
    
    phi = coord.Longitude(np.arctan2(t['Y_gal'], t['X_gal'])*u.rad - np.pi*u.rad).wrap_at(180*u.deg)
    lz = t['Lz'] / (-8*u.kpc*220*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    
    #print(t.colnames)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharex=True, sharey=True)
    
    
    plt.sca(ax[0])
    plt.plot(phi[ind_disk], lz[ind_disk], 'ko', mew=0, ms=4, alpha=0.5)
    #plt.plot(t['Vphi_gal'][ind_disk] +220, t['Vr_gal'][ind_disk], 'ko', mew=0, ms=2, alpha=0.5)
    
    plt.sca(ax[1])
    plt.plot(phi[ind_gse], lz[ind_gse], 'ko', mew=0, ms=4, alpha=0.5)
    #plt.plot(t['Vphi_gal'][ind_gse] +220, t['Vr_gal'][ind_gse], 'ko', mew=0, ms=2, alpha=0.5)
    
    plt.ylim(0.,3.)
    plt.xlim(-2,2)
    
    plt.tight_layout()
    


########
# APOGEE

def apogee_kiel():
    """"""
    t = Table(fits.getdata('/home/ana/data/apogee.fits'))
    print(t.colnames)

    plt.close()
    plt.figure()
    
    plt.plot(t['TEFF'], t['LOGG'], 'k.', ms=1, mew=0, alpha=0.2)
    
    plt.axhline(3.5, color='r')
    
    plt.xlim(7000,2000)
    plt.ylim(5,-1)
    plt.xlabel('$T_{eff}$ [K]')
    plt.ylabel('log g')
    
    plt.tight_layout()

def apogee_giants(Nstep=100, istep=0):
    """"""
    t = Table(fits.getdata('/home/ana/data/apogee.fits'))
    ind = (t['LOGG']<3.5) & np.isfinite(t['dist']) #& (t['Lz']>0.5)
    t = t[ind][istep::Nstep]
    #print(len(t))
    
    ceq = coord.SkyCoord(ra=t['RA']*u.deg, dec=t['DEC']*u.deg, distance=t['dist']*1e-3*u.kpc, pm_ra_cosdec=t['pmra']*u.mas/u.yr, pm_dec=t['pmdec']*u.mas/u.yr, radial_velocity=t['VHELIO_AVG']*u.km/u.s, frame='icrs')
    cgal = ceq.transform_to(coord.Galactocentric)
    w0 = gd.PhaseSpacePosition(cgal.cartesian)
    
    x = np.array([w0.pos.x.value, w0.pos.y.value, w0.pos.z.value]) * w0.pos.x.unit
    v = np.array([w0.vel.d_x.value, w0.vel.d_y.value, w0.vel.d_z.value]) * w0.vel.d_x.unit
    L = np.cross(x.value, v.value, axis=0) * w0.pos.x.unit * w0.vel.d_x.unit
    Ltot = np.linalg.norm(L.value, axis=0) * L.unit
    Lx = L[0]
    Ly = L[1]
    Lz = L[2]
    
    #Ek = w0.kinetic_energy().to(u.km**2*u.s**-2)
    #Epot = w0.potential_energy(ham.potential).to(u.km**2*u.s**-2)
    Etot = w0.energy(ham.potential).to(u.km**2*u.s**-2)
    
    ## circ setup from Rohan
    #def pot(R):
        #return gp.MilkyWayPotential().energy([R, 0, 0]*u.kpc).value[0]
    
    #pot_vec = np.vectorize(pot)
    
    #def Lcirc(Etot,R):
        #return -R*((2*(Etot - pot_vec(R)))**0.5)

    #def maxLcirc(Etot):
        #optfunc = functools.partial(Lcirc,Etot)
        #res = minimize(optfunc, np.array([0.1]), method='BFGS')
        #return np.abs(res.fun)

    #maxLcirc_vec = np.vectorize(maxLcirc)
    #maxLcirc_arr = maxLcirc_vec(np.linspace(-0.175, 0, 1000))
    
    #Lmax = np.interp(Etot.value/(1E+6), np.linspace(-0.175,0,1000), maxLcirc_arr)
    #circLz = np.nanmean(np.abs(Lz / Lmax)/1000.0)
    #circLtot = np.nanmean(np.abs(Ltot / Lmax)/1000.0)
    
    t['Lx'] = L[0].to(u.kpc**2/u.Myr)
    t['Ly'] = L[1].to(u.kpc**2/u.Myr)
    t['Lz'] = L[2].to(u.kpc**2/u.Myr)
    t['Lperp'] = np.sqrt(t['Lx']**2 + t['Ly']**2)
    t['E_tot_pot1'] = Etot.to(u.kpc**2*u.Myr**-2)
    #t['circLz_pot1'] = circLz
    #t['circLtot_pot1'] = circLtot
    
    ## calculate orbits
    #orbit = ham.integrate_orbit(w0, dt=1*u.Myr, n_steps=2)
    #try:
        #T = orbit.estimate_period()
    #except TypeError:
        #T = np.nan
    #t['orbit_period_pot1'] = T
    
    ## calculate actions
    #o = Orbit(ceq)
    
    ##giants = dict()
    #t['Jr'] = (o.jr(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    #t['Jz'] = (o.jz(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    #t['Jphi'] = (o.jp(pot=MWPotential2014) * u.kpc*u.km/u.s).to(u.kpc**2*u.Myr**-1)
    #t['Jtot'] = (t['Jr']**2 + t['Jz']**2 + t['Jphi']**2)**0.5
    
    
    t.write('../data/apogee_giants_{:d}.{:d}.fits'.format(Nstep, istep), overwrite=True)

def combine_apogee(tracer='giants', Nstep=100):
    """"""
    
    i = 0
    t = Table.read('../data/apogee_{:s}_{:d}.{:d}.fits'.format(tracer, Nstep, i))
    #t.remove_columns('orbit_period_pot1')
    
    for i in range(1,Nstep):
        t_ = Table.read('../data/apogee_{:s}_{:d}.{:d}.fits'.format(tracer, Nstep, i))
        #t_.remove_columns('orbit_period_pot1')
        t = vstack((t, t_))
    
    t.pprint()
    t.write('../data/apogee_{:s}.fits'.format(tracer), overwrite=True)

def apogee_elz(zmax=False):
    """"""
    t = Table(fits.getdata('../data/apogee_giants.fits'))
    #t = t[::5]
    if zmax:
        ind = np.abs(t['galz'])>2
        t = t[ind]
    print(t.colnames)
    print(np.nanmedian(t['Energy_err']), np.nanmedian(t['Lz_err']), np.nanmedian(t['dist_error']/t['dist']))
    
    plt.close()
    plt.figure()
    
    plt.plot(t['Lz'], t['E_tot_pot1'], 'k.', ms=1, mew=0, alpha=0.2)
    
    if not zmax:
        eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
        for e in eridge:
            plt.axhline(e, color='r', lw=0.2)
    
    plt.xlim(-6,6)
    plt.ylim(-0.3, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/apogee_elz_z.{:d}.png'.format(zmax))

def apogee_elz_pops(zmax=False, tracer='giants'):
    """"""
    t = Table(fits.getdata('../data/apogee_giants.fits'))
    #t = t[::5]
    if zmax:
        ind = (np.abs(t['galz'])>2)
        t = t[ind]
    print(t.colnames)
    
    ind_circular = (t['Lz']<0) & (t['e']<0.5) & (t['E_tot_pot1']>-0.16)
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.15)
    ind_ripple = (dlz>0.15) & (dlz<0.5)
    ind_radial = (dlz>0.5) & (dlz<1)
    
    #ind_ridge = (dlz<0.15)
    #ind_ripple = (dlz>0.15) & (dlz<0.7)
    #ind_radial = (dlz>0.7) & (dlz<1)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    
    plt.sca(ax[0])
    plt.plot(t['Lz'], t['E_tot_pot1'], 'k.', ms=1, mew=0, alpha=0.2)
    
    #for ind in [ind_ridge, ind_ripple, ind_radial]:
    for ind in [ind_ridge, ind_ripple]:
        plt.plot(t['Lz'][ind & ind_circular], t['E_tot_pot1'][ind & ind_circular], '.', ms=1)
    
    eridge = np.array([-0.146, -0.134, -0.127, -0.122, -0.116])
    for e in eridge:
        plt.axhline(e, color='r', lw=0.2)
    
    plt.xlim(-6,6)
    plt.ylim(-0.3, -0.02)
    
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    
    ebins = np.linspace(-0.16, -0.04, 100)
    plt.sca(ax[1])
    plt.hist(t['E_tot_pot1'][ind_circular], bins=ebins, histtype='step', density=True, color='k', alpha=0.2)
    for ind in [ind_ridge, ind_ripple]:
        plt.hist(t['E_tot_pot1'][ind & ind_circular], bins=ebins, histtype='step', density=True)
    
    for e in eridge:
        plt.axvline(e, color='r', lw=0.2)
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    plt.ylabel('Density [kpc$^{-2}$ Myr$^{2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/apogee_elz_ehist_z.{:d}.png'.format(zmax))

def apogee_afe(snr=10, zmax=0, tracer='giants'):
    """"""
    
    t = Table.read('../data/apogee_{:s}.fits'.format(tracer))
    #t = Table.read('../data/apogee_{:s}.fits'.format(tracer))
    #t = Table.read('../data/rcat_msto.fits')
    ind = (t['zmax']>zmax) #& (np.abs(t['galz'])>2)
    t = t[ind]
    
    ind_circular = (t['Lz']<0) & (t['e']<0.5) & (t['E_tot_pot1']>-0.16)
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    ind_radial = (dlz>1) & (dlz<1.5)
    
    ind_ridge = (dlz<0.15)
    ind_ripple = (dlz>0.15) & (dlz<0.7)
    ind_radial = (dlz>0.7) & (dlz<1)
    
    fehbins = np.linspace(-3,0.,40)
    afe = t['MG_H'] - t['FE_H']
    
    plt.close()
    fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
    
    for ind in [ind_ridge, ind_ripple, ind_radial]:
        plt.sca(ax[0])
        plt.hist(t['FE_H'][ind & ind_circular & (afe>0.25)], bins=fehbins, histtype='step', density=True)
        
        plt.sca(ax[1])
        plt.plot(t['FE_H'][ind & ind_circular], afe[ind & ind_circular], 'o', ms=1)
    
    plt.xlim(-3.5,0.5)
    plt.ylim(-0.2,0.6)
    
    plt.tight_layout()

def apogee_alfe(snr=10, zmax=0, tracer='giants'):
    """"""
    
    t = Table.read('../data/apogee_{:s}.fits'.format(tracer))
    #t = Table.read('../data/apogee_{:s}.fits'.format(tracer))
    #t = Table.read('../data/rcat_msto.fits')
    ind = (t['zmax']>zmax) & (np.abs(t['galz'])>2)
    t = t[ind]
    
    ind_circular = (t['Lz']<0) & (t['e']<0.5) & (t['E_tot_pot1']>-0.16)
    
    par = np.load('../data/elz_ridgeline_{:s}.npy'.format(tracer))
    poly = np.poly1d(par)
    dlz = t['Lz'] - poly(t['E_tot_pot1'])
    
    ind_ridge = (dlz<0.3)
    ind_ripple = (dlz>0.3) & (dlz<1)
    ind_radial = (dlz>1) & (dlz<1.5)
    
    ind_ridge = (dlz<0.15)
    ind_ripple = (dlz>0.15) & (dlz<0.7)
    ind_radial = (dlz>0.7) & (dlz<1)
    
    
    fehbins = np.linspace(-3,0.,40)
    alfe = t['AL_H'] - t['FE_H']
    afe = t['MG_H'] - t['FE_H']
    
    plt.close()
    fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)
    
    for ind in [ind_ridge, ind_ripple, ind_radial]:
        plt.sca(ax[0])
        plt.hist(t['FE_H'][ind & ind_circular & (afe>0.25)], bins=fehbins, histtype='step', density=True)
        
        plt.sca(ax[1])
        plt.plot(t['FE_H'][ind & ind_circular], alfe[ind & ind_circular], 'o', ms=1)
    
    plt.xlim(-3.5,0.5)
    plt.ylim(-0.6,0.6)
    
    plt.tight_layout()



def apogee_circ(zmax=10):
    """"""
    
    t = Table(fits.getdata('../data/apogee_giants.fits'))
    ind = (t['Lz']<-0.5) & (t['zmax']>zmax)
    t = t[ind]
    t = t[::1]
    #print(t.colnames)
    
    ceq = coord.ICRS(ra=t['RA']*u.deg, dec=t['DEC']*u.deg, distance=t['dist']*u.kpc)
    cg = ceq.transform_to(coord.Galactic)
    cgal = ceq.transform_to(coord.Galactocentric)
    
    ind = (np.abs(cg.b)>40*u.deg)
    ind = (np.abs(cgal.z)>2*u.kpc) & (np.abs(cg.b)>40*u.deg)
    ind = (np.abs(ceq.distance)>2*u.kpc) & (np.abs(cg.b)>40*u.deg) & (t['FE_H']<-0.7) & (t['FE_H']>-1)
    t = t[ind]
    
    eridge = np.array([-0.1475, -0.1326, -0.1262, -0.1194, -0.1129, -0.1061, -0.0962])
    eridge = np.array([-0.1455, -0.1326, -0.1269, -0.1201, -0.114, -0.1021, -0.0957])
    #print(eridge[1:-1]-eridge[2:])
    N = np.size(eridge)
    cridge = [mpl.cm.magma(x/N) for x in range(N)]
    climit = 0.4
    
    ebands = np.array([[-0.152, -0.142], [-0.1344, -0.1307], [-0.128, -0.124], [-0.1210, -0.1166], [-0.1139, -0.1109], [-0.1074, -0.1048], [-0.0976, -0.0921],])
    N = np.shape(ebands)[0]
    #print(N)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    
    plt.sca(ax[0])
    ind_ = t['circLz_pot1']>climit
    bins = np.linspace(-0.17,-0.06, 100)
    
    plt.hist(t['E_tot_pot1'][ind_], bins=bins, density=True, color='k', lw=2, histtype='step')
    
    plt.ylabel('Density')
    plt.text(0.05,0.85,'Z$_{{max}}$ = {:g} kpc'.format(zmax), transform=plt.gca().transAxes)

    plt.sca(ax[1])
    plt.plot(t['E_tot_pot1'], t['circLz_pot1'], 'ko', ms=2, mew=0, alpha=0.4)
    
    plt.axhline(climit, color='r')
    plt.xlim(-0.18,-0.05)
    
    for i in range(2):
        plt.sca(ax[i])
        for k, e in enumerate(eridge):
            plt.axvline(e, color=cridge[k], lw=0.5, zorder=0, alpha=0.5)
        
        for k in range(N):
            plt.axvspan(ebands[k][0], ebands[k][1], color=cridge[k], alpha=0.2)
    
    plt.xlabel('$E_{tot}$ [kpc$^2$ Mpc$^{-2}$]')
    plt.ylabel('Circularity')
    plt.ylim(0,1)
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/apogee_ehist_zmax.{:02d}.png'.format(zmax))

def zmax():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>3) & (t['Lz']<0) & (t['circLz_pot1']>0.4)
    t = t[ind]
    
    ta = Table(fits.getdata('../data/apogee_giants.fits'))
    ind = (ta['Lz']<0)
    ta = ta[ind]
    
    #ceq = coord.ICRS(ra=ta['RA']*u.deg, dec=ta['DEC']*u.deg)
    #cg = ceq.transform_to(coord.Galactic)
    #ind = (np.abs(cg.b)>40*u.deg)
    #ta = ta[ind]
    
    print(np.sum(np.abs(t['Z_gal'])>2)/len(t), np.sum(np.abs(t['Z_gal'])>2))
    print(np.sum(np.abs(ta['galz'])>2)/len(ta), np.sum(np.abs(ta['galz'])>2))
    
    
    bins = np.linspace(0,10,100)
    
    plt.close()
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    
    plt.sca(ax[0])
    plt.hist(t['zmax_pot1'], bins=bins, density=True, histtype='step', label='H3')
    plt.hist(ta['zmax'], bins=bins, density=True, histtype='step', label='APOGEE')
    
    plt.xlabel('$z_{max}$ [kpc]')
    plt.ylabel('Density [kpc$^{-1}$]')
    
    plt.sca(ax[1])
    plt.hist(np.abs(t['Z_gal']), bins=bins, density=True, histtype='step', label='H3')
    plt.hist(np.abs(ta['galz']), bins=bins, density=True, histtype='step', label='APOGEE')
    
    plt.xlabel('$z_{max}$ [kpc]')
    plt.ylabel('Density [kpc$^{-1}$]')
    
    plt.legend()
    
    plt.tight_layout()

def apogee_periods():
    """"""
    t = Table(fits.getdata('../data/apogee_giants.fits'))
    print(len(t))
    ind = (t['Lz']<-0.5) #& (t['zmax']>zmax)
    t = t[ind]
    #t = t[::5]
    print(len(t))
    
    Ta = 2*np.pi/t['omega_r']*u.Gyr
    #Ta = 1/t['omega_r']*u.Gyr
    Ta = (t['orbit_period_pot1']*u.Myr).to(u.Gyr)
    
    # H3
    t = Table.read('../data/rcat_giants.fits')
    ind_circ = (t['circLz_pot1']>0.35) & (t['Lz']<0)
    t = t[ind_circ]
    
    Th = (t['orbit_period_pot1']*u.Myr).to(u.Gyr)
    
    bins = np.linspace(0,0.5,100)
    bins_h3 = np.linspace(0,0.5,100)
    
    plt.close()
    plt.figure(figsize=(15,6))
    
    plt.hist(Ta.value, bins=bins, density=True, histtype='step')
    plt.hist(Th.value, bins=bins_h3, density=True, histtype='step')
    
    plt.xlim(0.04,0.5)
    plt.tight_layout()

def apogee_rgal():
    """"""
    t = Table(fits.getdata('../data/apogee_giants.fits'))
    #print(t.colnames)
    ind = (t['Lz']<-0.5) #& (t['zmax']>zmax)
    t = t[ind]
    
    plt.close()
    plt.figure()
    
    plt.hist(t['galr'], bins=np.linspace(0,5,50))
    
    plt.tight_layout()
    
    

def sgr():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    print(t.colnames)
    
    ind = (t['Sgr_l']>50) & (t['Sgr_l']<70) & (t['dist_adpt']>74) & (t['dist_adpt']<92)
    #ind = (t['Sgr_l']>282) & (t['Sgr_l']<292) & (t['dist_adpt']>70) & (t['dist_adpt']<76)
    ind_sgr = t['Sgr_FLAG']==1
    
    print(t['Vrad'][ind])
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.plot(t['Sgr_l'], t['dist_adpt'], 'k.', ms=2)
    plt.plot(t['Sgr_l'][ind_sgr], t['dist_adpt'][ind_sgr], 'ro', ms=2)
    
    plt.ylim(5,130)
    
    plt.tight_layout()

def fan():
    """"""
    t = Table.read('../data/rcat_giants.fits')
    ind = (t['SNR']>10)
    t = t[ind]
    
    plt.close()
    plt.figure(figsize=(13,8))
    
    #plt.scatter(t['Lz'], t['E_tot_pot1'], s=0.08*t['SNR']**1.5, c='0.1')
    #plt.scatter(t['Lz'], t['E_tot_pot1'], s=0.05*t['SNR']**1.7, c='0.1')
    plt.plot(t['Lz'], t['E_tot_pot1'], 'ko', mew=0, ms=3)
    
    plt.xlim(-6,6)
    plt.ylim(-0.18, -0.02)
    plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig('../plots/elz_fan.svg')
