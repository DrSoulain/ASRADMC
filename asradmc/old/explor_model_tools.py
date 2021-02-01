#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:01:24 2020

@author: asoulain
"""

import os
import pickle
import sys

import ffmpeg
import numpy as np
from astropy import constants as const
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from numpy.lib.npyio import save
from scipy import ndimage
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from uncertainties import ufloat
from uncertainties.umath import atan

from astools.all import AllMyFields
from asradmc.Radmc_utils import Radmc_rho_save

ms = const.M_sun.cgs.value     # Solar mass          [g]
au = const.au.cgs.value    # Astronomical Unit       [cm]

# print(1e-7*0.01*ms/const.M_earth.cgs.value)

if sys.version[0] == '2':
    from asradmc.ramses_tools import RamsesGazToDust

fs = 18


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception:
            print('pb', pickle_file)
            return None
    return pickle_data


def model_linear(x, a, b):
    y = a * x + b
    return y


def model_pow(x, a, b):
    y = a * x**b
    return y


def model_power(x, a, b, c, d):

    y = a * x**3 + b * x**2 + c * x + d
    return y


def extract_prf(im, pos, num=1000, display=False, p=.2, log=True, kernel=1, vmin=2e-5, vmax=0.2, origin=[0, 0],
                pixel=1):
    n = len(im)
    p0 = pos[0]
    p1 = pos[1]
    x1, y1 = p1[1], p1[0]  # n-1, 0 # These are in _pixel_ coordinates!!
    x0, y0 = p0[1], p0[0]  # 0, n-1

    y_max = np.max([x0, x1])
    x_max = np.max([y0, y1])
    y_min = np.min([x0, x1])
    x_min = np.min([y0, y1])

    dx = x_max - x_min
    dy = y_max - y_min

    x_mid = x_min + dx / 2.
    y_mid = y_min + dy / 2.

    x_WR = origin[0]
    y_WR = origin[1]

    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    pix = 22.6 / n

    image = np.rot90(im)
    zi = image[x.astype(np.int), y.astype(np.int)]

    xi = np.linspace(0, d, num) * pix

    cond = zi >= 1e-50

    X, Y = xi[cond], medfilt(zi[cond], kernel)

    rnuc = np.sqrt((x_WR - x_mid)**2 + (y_WR - y_mid)**2)
    rnuc_mas = rnuc * pixel

    x = np.arange(n)
    y = np.arange(n)
    xy = np.meshgrid(np.arange(n), np.arange(n))[0]
    xy2 = np.meshgrid(np.arange(n), np.arange(n))[1]

    s1 = 100
    s2 = 150
    im_left = image.copy()
    im_left[xy > s1] = 0
    im_left[xy2 > s2] = 0

    im_right = image.copy()
    im_right[xy <= s1] = 0
    im_right[xy2 > s2] = 0

    m_left = np.where(im_left == np.max(im_left))
    m_right = np.where(im_right == np.max(im_right))

    y_max = np.max([m_right[0], m_left[0]])
    x_max = np.max([m_right[1], m_left[1]])
    y_min = np.min([m_right[0], m_left[0]])
    x_min = np.min([m_right[1], m_left[1]])

    dx = x_max - x_min
    dy = y_max - y_min

    x_mid2 = x_min + dx / 2.
    y_mid2 = y_min + dy / 2.

    rnuc2 = np.sqrt((x_WR - x_mid2)**2 + (y_WR - y_mid2)**2)
    rnuc_mas2 = rnuc2 * pixel

    w1 = ((m_right[1] - m_left[1])**2 + (m_right[0] - m_left[0])**2)**0.5
    w1_pix = w1 * pixel

    alpha_lin = 2 * np.rad2deg(np.arctan(w1_pix / (2 * rnuc_mas2)))

    vgrad = np.gradient(np.rot90(im))
    mag = np.sqrt(vgrad[0]**2 + vgrad[1]**2)
    lap_grad = (ndimage.laplace(mag, output=None, mode='reflect', cval=1))
    lap = ndimage.laplace(im, output=None, mode='reflect', cval=0)

    mag_l = mag.copy()
    mag_l[xy > n // 2] = 0
    mag_l[xy2 > n // 2 + 10] = 0

    mag_r = mag.copy()
    mag_r[xy < n // 2] = 0

    mag_r[im_right < im_right.max() / 1.5] = 0
    mag_r[xy2 > n // 2] = 0

    m_left2 = np.where(mag_l == np.max(mag_l))
    m_right2 = np.where(mag_r == np.max(mag_r))

    y_max = np.max([m_right2[0], m_left2[0]])
    x_max = np.max([m_right2[1], m_left2[1]])
    y_min = np.min([m_right2[0], m_left2[0]])
    x_min = np.min([m_right2[1], m_left2[1]])

    dx = x_max - x_min
    dy = y_max - y_min

    x_mid2 = x_min + dx / 2.
    y_mid2 = y_min + dy / 2.

    rnuc3 = np.sqrt((x_WR - x_mid2)**2 + (y_WR - y_mid2)**2)
    rnuc_mas3 = rnuc3 * pixel

    w1_3 = ((m_right2[1] - m_left2[1])**2 + (m_right2[0] - m_left2[0])**2)**0.5
    w1_pix3 = w1_3 * pixel

    alpha_lin3 = 2 * np.rad2deg(np.arctan(w1_pix3 / (2 * rnuc_mas3)))

    corr_lap = (lap_grad) - lap

    corr_lap[lap > 0] = 0
    corr_lap[corr_lap < 0] = 0

    if display:
        plt.figure()
        plt.imshow(image, origin='upper', norm=PowerNorm(
            p), cmap='afmhot', vmax=vmax, vmin=vmin)
        plt.plot([y0, y1], [x0, x1], 'w--', lw=1)
        plt.plot(y1, x1, 'ro')
        plt.plot(y0, x0, 'co')
        # plt.plot(x_mid, y_mid, 'ro')
        plt.plot(x_mid2, y_mid2, 'go')
        plt.plot(x_WR, y_WR, 'w*')
        plt.plot(m_left[1], m_left[0], 'r+')
        plt.plot(m_right[1], m_right[0], 'r+')
        plt.plot(m_right2[1], m_right2[0], 'b+')
        plt.plot(m_left2[1], m_left2[0], 'b+')

        plt.figure()
        plt.plot(X, Y)
        if log:
            plt.yscale('log')

    return X, Y, rnuc_mas, alpha_lin3, alpha_lin


def extract_edge(im, mix, fov=21.6, side=1, orient='h', display=False):
    npts = len(im)
    xy = np.meshgrid(np.arange(npts), np.arange(npts))[0]
    xy2 = np.meshgrid(np.arange(npts), np.arange(npts))[1]

    if orient == 'h':
        tab_xy = xy2
    else:
        tab_xy = xy

    im_left = im.copy()
    l_max = 105
    im_left[tab_xy > npts // 2 - l_max] = 0
    im_right = im.copy()
    r_max = 80
    im_right[tab_xy <= npts // 2 + r_max] = 0

    if mix <= 20:
        rmax2 = 100
    else:
        rmax2 = 200

    rmax2 = 200
    lap_right = ndimage.laplace(im_right)
    lap_right[tab_xy <= npts // 2 + r_max + 2] = 0

    lap_right[xy > rmax2] = 0
    if (mix < 10) & (mix >= 5):
        lim_r = 1e-16
    elif (mix < 5) & (mix >= 1):
        lim_r = 2e-16
    elif (mix < 1):
        lim_r = 2.5e-16
    else:
        lim_r = 4e-17
    edge_right = np.where(lap_right > lim_r)

    lap_left = ndimage.laplace(im_left)
    lap_left[tab_xy > npts // 2 - l_max - 2] = 0

    if mix < 7:
        lim_l = 1.2e-16
    else:
        lim_l = 3e-17
    edge_left = np.where(lap_left > lim_l)

    if display:
        plt.figure()
        plt.title(r'Edge detection: $\xi$  = %2.2f%%' % mix)
        plt.imshow(lap_left + lap_right)
        plt.plot(edge_right[1], edge_right[0],
                 color='orange', marker='+', ls='')
        plt.plot(edge_left[1], edge_left[0], color='orange', marker='+', ls='')

    return im_left, im_right, edge_left, edge_right


def Open_cube(modeldir, xi, rnuc, mix, amr=True, npix=256):

    if amr:
        s_amr = 'Amr'
    else:
        s_amr = 'NoAmr'

    imdir = modeldir + \
        'Saved_images/Model_fitSED/rnuc=%2.2f/xi=%2.2f/' % (rnuc, xi)

    imfile = imdir + \
        'ramses_cube_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_incl=0_posang=0.00_%s' % (
            npix, xi, rnuc, mix, s_amr)

    dic = load_pickle(imfile + '.dpy')

    data = dic['cube']
    wl = dic['wl']
    flux = dic['flux']
    extent = dic['extent']

    band = {'0.55': 'V',
            '1.21': 'J',
            '1.65': 'H',
            '2.27': 'K',
            '3.5': 'L',
            '10.0': 'N',
            '300.0': 'band3'}

    cube = {}
    for i in range(len(data)):
        cube[band[str(wl[i])]] = {'im': data[i], 'wl': wl[i], 'f': flux[i]}
    cube['extent'] = extent
    cube['xi'] = xi
    cube['mix'] = mix
    cube['rnuc'] = rnuc

    return AllMyFields(cube)


def norm(tab):
    return tab / np.max(tab)


def spiral_archimedian(t, param):
    a = param['d_arm'] / (2 * np.pi) / param['pixel_size']
    eta = np.deg2rad(param['eta'])  # angle sky
    theta = np.deg2rad(param['theta'])  # angle incl x
    phi = np.deg2rad(param['phi'])   #
    x = (a * t) * (np.cos(t + eta))
    y = (a * t) * (np.sin(t + eta))

    # rotation inclinaison
    xp = x
    yp = y * np.cos(theta)

    # Rotation projection
    xpp = xp * np.cos(phi) - yp * np.sin(phi) + param['x0']
    ypp = xp * np.sin(phi) + yp * np.cos(phi) + param['y0']
    return xpp, ypp


def curvilinear_flux(image, turn, param, npts=200):
    '''
    Compute curvilinear flux along the archimedian spiral.
    '''
    t = np.linspace(0, turn * 2 * np.pi, npts)
    ui, vi = spiral_archimedian(t, param)
    flux = []
    x = np.arange(len(image))
    y = np.arange(len(image))
    f = interp2d(x, y, image, 'cubic')
    for xx in np.arange(len(ui)):
        flux.append(f(ui[xx], vi[xx])[0])

    flux = np.array(flux)

    flux[flux < 0] = 1e-10

    err_flux_rel = np.sqrt(flux) / flux

    flux_norm = 100 * norm(np.array(flux))

    err_flux = err_flux_rel * flux_norm
    return flux_norm, err_flux, [ui, vi]


def Circle_profil(l_r, image, pix_mas, x0, y0, verbose=False, display=False):

    npts = len(image)

    tmin = -0.5
    tmax = np.pi / 1.2

    t2 = np.linspace(tmax, tmin, 300)

    l_frac, l_theta = [], []

    for r in l_r:
        x_ring = r * np.cos(t2) / pix_mas + x0  # + 20
        y_ring = r * np.sin(t2) / pix_mas + y0

        x = np.arange(npts)
        y = np.arange(npts)
        f = interp2d(x, y, image, 'cubic', copy=False,
                     bounds_error=False, fill_value=0)

        flux = []
        for xx in xrange(len(x_ring)):
            flux.append(f(x_ring[xx], y_ring[xx])[0])
        flux = np.array(flux)
        flux[flux < 1e-20] = 1e-22

        cond_dust = flux > 1e-19

        t_l = t2[cond_dust][0]
        t_r = t2[cond_dust][-1]
        f_l = flux[cond_dust][0]
        f_r = flux[cond_dust][-1]

        cond_large = (t2 <= t_l) & (t2 >= t_r)

        n_thickness = float(len(flux[cond_large]))
        n_thickness_dust = float(len(flux[cond_dust]))

        pix_t = abs(np.diff(t2)[0])
        t_thickness = round(np.rad2deg(n_thickness * pix_t), 2)
        t_thickness_dust = round(np.rad2deg(n_thickness_dust * pix_t), 2)

        frac_dust = 100. * t_thickness_dust / t_thickness

        l_frac.append(frac_dust)
        l_theta.append(t_thickness)

        if verbose:
            print(r'r = %2.1f mas, theta = %2.1f deg, filling factor = %2.1f %%' % (
                r, t_thickness, frac_dust))

        savedict = {'t2': t2, 'flux': flux,
                    'cond_large': cond_large, 'cond_dust': cond_dust,
                    't_l': t_l, 't_r': t_r, 'f_l': f_l,
                    'f_r': f_r}

        if display:
            file = open('saveprf%s.dpy' % r, 'wb')
            pickle.dump(savedict, file, 2)
            file.close()
            print('save')
            vmax = image.max()
            vmin = vmax / 1e4
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.semilogy(t2, flux, color='lightgrey', lw=1,
                         label='Interpolated density')
            plt.semilogy(t2[cond_large], flux[cond_large], 'o',
                         color='cadetblue', label='Inside spiral')
            plt.semilogy(t2[cond_dust], flux[cond_dust], '+', color='goldenrod',
                         alpha=.8, label='Dust (%2.1f%%)' % frac_dust)
            plt.semilogy([t_l, t_r], [f_l, f_r], 's', color='crimson',
                         label=r'Thickness (%2.1f$\degree$)' % t_thickness)
            plt.legend(fontsize=10, loc=3)
            plt.xlim(tmax, tmin)
            plt.subplot(1, 2, 2)
            plt.imshow(image, norm=LogNorm(),
                       cmap='gist_earth', vmin=vmin, vmax=vmax)
            plt.plot(x_ring, y_ring, color='lightyellow',
                     label='Extracted profil (r = %2.1f mas)' % (r))
            plt.legend(fontsize=10, loc=2)

            plt.axis(np.array([0, npts, npts, 0]) - 0.5)
            plt.subplots_adjust(top=0.968,
                                bottom=0.065,
                                left=0.05,
                                right=0.988,
                                hspace=0.2,
                                wspace=0.002)

    return np.array(l_frac), np.array(l_theta)


def explor_Tsub(npix, rnuc, xi, mix, modeldir, amr=True, Tsub=2000, Tm=3500, Tstep=100, display=False):

    if amr:
        s_amr = 'AMR'
    else:
        s_amr = 'NoAmr'

    Tfile = modeldir + 'Saved_temp/rnuc=%2.2f/xi=%2.2f/' % (rnuc, xi) + \
        'Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s' % (
            npix, xi, rnuc, mix, s_amr)
    Rhofile = modeldir + \
        'Saved_dens/Rhodist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s' % (
            npix, xi, rnuc, mix, s_amr)

    dic_T = load_pickle(Tfile + '.dpy')

    hdu = fits.open(Tfile + '.fits')
    Tdist = hdu[0].data

    cond = Tdist >= 0

    c1 = Tdist == 0

    print(100 * float(len(Tdist[c1])) / len(Tdist))

    try:
        dic_rho = load_pickle(Rhofile + '.dpy')

        rho = dic_rho['rho']
        Vol_cell = dic_rho['Vol_cell']
        mdust_tot = dic_rho['mass_tot']

        cond_sub = Tdist >= Tsub

        rho_hot = rho[cond_sub]
        V_hot = Vol_cell[cond_sub]

        m_dust_hot = (rho_hot * V_hot).sum() / ms

        frac_hot_dust = 100 * m_dust_hot / mdust_tot

        mhot_terre = frac_hot_dust * mdust_tot / 100. * ms  # /const.M_earth.cgs.value

    except Exception:
        frac_hot_dust = np.nan
        pass

    Thot = Tdist[Tdist >= Tsub]
    Thot_m = np.mean(Thot)

    n_over_Tsub = dic_T['Nhot']
    n_dust = float(len(Tdist[cond]))
    n_over_rel = 100 * (n_over_Tsub / n_dust)
    Tmean = np.mean(Tdist[cond])
    Tmax = dic_T['Tmax']
    try:
        dustmass = dic_T['param']['dustmass']
    except KeyError:
        dustmass = 0

    if display:
        plt.figure()
        plt.hist(Tdist[cond & (Tdist < Tsub)], bins=np.arange(0, Tm, Tstep))
        plt.hist(Tdist[Tdist >= Tsub], bins=np.arange(0, Tm, Tstep), color='crimson', alpha=.5,
                 label='Hot cells = %i (%2.3f%%)' % (n_over_Tsub, 100 * (n_over_Tsub / n_dust)))

        plt.axvline(Tmean, label=r'T$_{mean}$ (%2.0f K)' % Tmean)
        plt.axvline(Tsub, linestyle='--',
                    color='r', label=r'T$_{sub}$ (%2.0f K)' % Tsub)
        plt.legend(loc='best', fontsize=8)
        plt.xlabel("T [K]", fontsize=12, color='gray')
        plt.ylabel("# cells", fontsize=12, color='gray')
        plt.yscale('log')
        plt.ylim(1, 1e9)
        plt.xlim(0, 3500)
        plt.tight_layout()

    return n_over_Tsub, n_over_rel, dustmass, Tmax, Thot_m


def compute_alpha_side0(rho_dust, mix, l_max=20, display=True):

    im = np.rot90(rho_dust.sum(axis=0))

    npts = len(im)
    xy = np.meshgrid(np.arange(npts), np.arange(npts))[0]
    xy2 = np.meshgrid(np.arange(npts), np.arange(npts))[1]

    orient = 'h'
    if orient == 'h':
        tab_xy = xy2
    else:
        tab_xy = xy
    im_left = im.copy()
    im_left[tab_xy > npts // 2 - l_max] = 0

    im_right = im.copy()
    r_max = l_max
    im_right[tab_xy <= npts // 2 + r_max] = 0

    lap_right = ndimage.laplace(im_right)
    lap_left = ndimage.laplace(im_left)

    lap_right[xy < npts // 2] = 0
    lap_left[xy < npts // 2] = 0
    lap_left[tab_xy > npts // 2 - l_max - 2] = 0
    lap_right[tab_xy <= npts // 2 + r_max + 2] = 0

    xx, yy = np.meshgrid(np.arange(npts) - npts // 2,
                         np.arange(npts) - npts // 2)
    dist = np.sqrt(xx**2 + yy**2)

    if (mix < 10) & (mix >= 5):
        lim_r = 2e-16
        dr = 70
    elif (mix < 5) & (mix >= 3):
        lim_r = 3e-16
        dr = 70
    elif (mix < 3) & (mix >= 1):
        lim_r = 3e-16
        dr = 80
    elif (mix < 1):
        dr = 80
        lim_r = 2.5e-16
    else:
        dr = 65
        lim_r = 1e-16

    # plt.figure()
    # plt.imshow(dist)
    lap_right[dist < dr] = 0
    lap_left[dist < dr] = 0

    edge_right = np.where(lap_right > lim_r)
    edge_left = np.where(lap_left > lim_r)

    xl, yl = list(edge_left[1]), list(edge_left[0])
    xr, yr = list(edge_right[1]), list(edge_right[0])
    xl.append(231)
    yl.append(126)
    xr.append(231)
    yr.append(126)

    try:
        popt_u, pcov_u = curve_fit(model_linear, xl, yl)
        popt_d, pcov_d = curve_fit(model_linear, xr, yr)

        p_sigma_u = np.sqrt(np.diag(pcov_u))
        p_sigma_d = np.sqrt(np.diag(pcov_d))

        a_u = popt_u[0]
        a_d = popt_d[0]

        e_a_u = ufloat(a_u, p_sigma_u[0])
        e_a_d = ufloat(a_d, p_sigma_d[0])

        a_rad = atan((e_a_d - e_a_u) / (1 + (e_a_u * e_a_d)))
        a_deg = (a_rad * 180. / np.pi)
        if a_deg > 0:
            a_deg = 180 - a_deg
        else:
            a_deg = -a_deg

        x_mod = np.linspace(-150, 250, 100)
        y_mod_u = model_linear(x_mod, *popt_u)
        y_mod_d = model_linear(x_mod, *popt_d)
    except TypeError:
        a_deg = ufloat(0, 0)

    if display:
        plt.figure()
        plt.clf()
        plt.imshow(im, cmap='gist_earth')
        plt.plot(xl, yl, 'w+')
        plt.plot(xr, yr, 'w+')
        try:
            plt.plot(x_mod, y_mod_u, '--', color='crimson')
            plt.plot(x_mod, y_mod_d, '--', color='crimson')
        except:
            pass
        plt.axis([0, len(im), len(im), 0])
        plt.tight_layout()

        lap_left[lap_left < 0] = 0
        lap_right[lap_right < 0] = 0
        plt.figure()
        plt.title(r'Edge detection: $\xi$  = %2.2f%%' % mix)
        plt.imshow(lap_left + lap_right, cmap='gist_earth')
        plt.colorbar()
        plt.plot(xr, yr,
                 color='orange', marker='+', ls='')
        plt.plot(xl, yl,
                 color='orange', marker='+', ls='')
    print(a_deg)
    return a_deg


def Compute_alpha_vs_mix(l_mix, dust_param, dict_simu, fov=0.0216, display=False, save=False):

    x_mod = np.linspace(-150, 250, 100)

    l_alpha_side, l_alpha_e = [], []

    j = 0
    for i_m in l_mix:
        dust_param['mix'] = i_m / 100.
        rho_dust, gridinfo, dust_mass = RamsesGazToDust(
            dict_simu, dust_param, display=False)

        alpha0 = compute_alpha_side0(rho_dust, i_m, display=display)

        im_dens_sum = np.rot90(rho_dust.sum(axis=1))

        A, B, e_u, e_d = extract_edge(im_dens_sum, i_m,
                                      orient='h', display=display)

        popt_u, pcov_u = curve_fit(model_linear, e_u[1], e_u[0])
        popt_d, pcov_d = curve_fit(model_linear, e_d[1], e_d[0])

        p_sigma_u = np.sqrt(np.diag(pcov_u))
        p_sigma_d = np.sqrt(np.diag(pcov_d))

        a_u = popt_u[0]
        a_d = popt_d[0]

        e_a_u = ufloat(a_u, p_sigma_u[0])
        e_a_d = ufloat(a_d, p_sigma_d[0])

        a_rad = atan((e_a_d - e_a_u) / (1 + (e_a_u * e_a_d)))
        a_deg = a_rad * 180. / np.pi

        l_alpha_side.append(a_deg.nominal_value)
        l_alpha_e.append(a_deg.std_dev)

        y_mod_u = model_linear(x_mod, *popt_u)
        y_mod_d = model_linear(x_mod, *popt_d)

        dpc = 2580
        npts = 256
        extent = (np.array([-fov / 2., fov / 2, -fov / 2, fov / 2])) * dpc
        pix_size = (2 * abs(extent[0])) / npts

        x_edge_u, y_edge_u = pix_size * \
            (e_u[1] - npts // 2), pix_size * (e_u[0] - npts // 2)
        x_edge_d, y_edge_d = pix_size * \
            (e_d[1] - npts // 2), pix_size * (e_d[0] - npts // 2)
        x_mod_c = pix_size * (x_mod - npts // 2)
        y_mod_u_c = pix_size * (y_mod_u - npts // 2)
        y_mod_d_c = pix_size * (y_mod_d - npts // 2)
        if display:
            plt.figure(20)
            plt.clf()
            plt.imshow(im_dens_sum, cmap='gist_earth')
            plt.plot(e_u[1], e_u[0], 'w+')
            plt.plot(e_d[1], e_d[0], 'w+')
            plt.plot(x_mod, y_mod_u, '--', color='crimson')
            plt.plot(x_mod, y_mod_d, '--', color='crimson')
            plt.title(r'Side view: mix = %2.0f%%, $\alpha_2$ = %2.1f $\degree$' % (
                i_m, a_deg.nominal_value))
            plt.axis([0, len(im_dens_sum), len(im_dens_sum), 0])
            plt.tight_layout()

            plt.figure(21, figsize=(8, 7))
            plt.imshow(im_dens_sum, cmap='gist_earth', extent=extent,
                       origin='lower')
            plt.xlabel('X [AU]', fontsize=fs)
            plt.ylabel('Y [AU]', fontsize=fs)
            plt.plot(x_edge_u, y_edge_u, '+', color='w',
                     ms=10, label='Edges detection')
            plt.plot(x_edge_d, y_edge_d, '+', color='w', ms=10)
            # plt.plot(e_d[1], e_d[0], 'w+')
            plt.plot(x_mod_c, y_mod_u_c, '--', lw=4,
                     color='crimson', label='fit', )
            plt.plot(x_mod_c, y_mod_d_c, '--', lw=4, color='crimson')

            plt.tick_params(left='off', bottom='off')

            # plt.axvspan(-70, extent[0], color='black')
            # plt.title(r'Side view: mix = %2.0f%%, $\alpha_2$ = %2.1f $\degree$' % (
            # i_m, a_deg.nominal_value))
            plt.ylim(extent[0], extent[1])
            plt.xlim(extent[0], extent[1])
            # plt.legend(loc=2, fontsize=fs)
            plt.subplots_adjust(right=1, left=0.07, bottom=0.09, top=1,)

            if save:
                if j <= 9:
                    plt.savefig('anim_alpha_mix/im00%i.png' %
                                j)  # , dpi = 300)
                elif j <= 99:
                    plt.savefig('anim_alpha_mix/im0%i.png' % j)  # , dpi = 300)
                else:
                    plt.savefig('anim_alpha_mix/im%i.png' % j)  # , dpi = 300)
            j += 1

    if save:
        os.system('rm anim_alpha_mix/alpha_vs_mix.gif')
        os.system('rm anim_alpha_mix/alpha_vs_mix.mp4')
        (
            ffmpeg.input('anim_alpha_mix/im*.png',
                         pattern_type='glob', framerate=10)
            .output('anim_alpha_mix/alpha_vs_mix.gif')
            .run()
        )
        (
            ffmpeg.input('anim_alpha_mix/*.png',
                         pattern_type='glob', framerate=15)
            .output('anim_alpha_mix/alpha_vs_mix.mp4')
            .run()
        )

    l_alpha = np.array(l_alpha_side)
    l_alpha_e = np.array(l_alpha_e)
    l_alpha = np.array([alpha0.nominal_value])
    l_alpha_e = np.array([alpha0.std_dev])

    return rho_dust, gridinfo, dust_mass, l_alpha, l_alpha_e


def Calc_param_hydro(mix, dict_simu, dust_param, dpc=2580, nturn=0.6,
                     step=0.066, eta1=180, save=False, verbose=True, display=True):

    npts = dict_simu['mix'].shape[0]

    fov = dict_simu['info']['fov'] / 1000.  # [arcsec]

    # pix_cm = fov * dpc * au / npts
    # pix_au = fov * dpc / npts
    pix_mas = (fov * 1000.) / npts

    unit = 'pix'

    if unit == 'mas':
        extent = np.array([-fov / 2., fov / 2, -fov / 2, fov / 2]) * 1000
        pix_size = (2 * abs(extent[0])) / npts
    elif unit == 'au':
        extent = (np.array([-fov / 2., fov / 2, -fov / 2, fov / 2])) * dpc
        pix_size = (2 * abs(extent[0])) / npts
    elif unit == 'cm':
        extent = (np.array([-fov / 2., fov / 2, -fov / 2, fov / 2])) * dpc * au
        pix_size = (2 * abs(extent[0])) / npts
    else:
        extent = np.array([-npts / 2., npts / 2, -npts / 2, npts / 2])
        pix_size = (2 * abs(extent[0])) / npts

    X_WR = dict_simu['info']['p_WR'][0] - 0.5
    Y_WR = npts - dict_simu['info']['p_WR'][1] - 0.5

    sep = dict_simu['info']['sep']
    bin_phase = dict_simu['info']['bin_phase']

    x_OB = sep * np.cos(np.deg2rad(-bin_phase))
    y_OB = sep * np.sin(np.deg2rad(-bin_phase))

    X_OB = X_WR + x_OB / ((fov * 1000.) / npts) + 1
    Y_OB = Y_WR - y_OB / ((fov * 1000.) / npts)

    X_OB, Y_OB = X_OB * pix_size, Y_OB * pix_size
    X_WR, Y_WR = X_WR * pix_size, Y_WR * pix_size

    rho_dust, gridinfo, dust_mass, alpha, e_alpha = Compute_alpha_vs_mix(
        [mix], dust_param, dict_simu, display=display)

    i_t = npts // 2
    # im_gaz = np.rot90(gridinfo.gaz_grid[:,:,i_t])
    # im_mix = np.rot90(gridinfo.mix[:,:,i_t])
    im_dens = np.rot90(rho_dust[:, :, i_t])
    im_dens_sum = np.rot90(rho_dust.sum(axis=2))

    cond_extrapol = im_dens_sum < 1e-19

    im_dens_sum[cond_extrapol] = 1e-50
    im_dens[cond_extrapol] = 1e-50

    l_r = np.linspace(dust_param['rnuc'] * 1e3, 15, 3)

    l_frac, l_theta = Circle_profil(
        l_r, im_dens, pix_mas, X_WR, Y_WR, verbose=verbose, display=False)

    frac = ufloat(np.mean(l_frac), np.std(l_frac))
    theta = ufloat(np.mean(l_theta), np.std(l_theta))
    # theta_rad = theta * np.pi/180.

    # e_thick = 2*rnuc*tan(theta_rad/2.)

    if verbose:
        print('\nFilling_factor = %2.1f +/- %2.1f %%' %
              (frac.nominal_value, frac.std_dev))
        print('Theta = %2.1f +/- %2.1f deg' %
              (theta.nominal_value, theta.std_dev))
        print('Alpha = %2.1f +/- %2.1f deg' % (alpha, e_alpha))
        # print('e_arm = %2.1f +/- %2.1f mas'%(e_thick.nominal_value, e_thick.std_dev))

    angle = theta.nominal_value / 2.
    angle2 = (1 - (frac.nominal_value / 100.)) * angle
    angle3 = alpha / 2.
    angle4 = (1 - (frac.nominal_value / 100.)) * angle3

    x_schem = np.linspace(0, 300, 10)
    coef1 = np.sin(np.deg2rad(angle)) / np.cos(np.deg2rad(angle))
    coef2 = np.sin(np.deg2rad(angle2)) / np.cos(np.deg2rad(angle2))
    coef3 = np.sin(np.deg2rad(angle3)) / np.cos(np.deg2rad(angle3))
    coef4 = np.sin(np.deg2rad(angle4)) / np.cos(np.deg2rad(angle4))

    y_schem = coef1 * x_schem
    y_schem2 = coef2 * x_schem
    y_schem3 = coef3 * x_schem
    y_schem4 = coef4 * x_schem

    open_angle = 2 * alpha  # theta.nominal_value#alpha
    # eta1 = 180
    eta2 = eta1 + open_angle
    eta3 = eta1 + theta.nominal_value

    param_spiral = {
        'eta': eta1,  # 181.5            #rotation phase
        'theta': 180,  # 2d projection angle
        'phi': 0,  # Inclinaison line of sight
        'x0': X_WR + 20,
        'y0': Y_WR,
        'd_arm': step * 1e3,
        'pixel_size': fov / npts * 1000.}

    param_spiral2 = param_spiral.copy()
    param_spiral2['eta'] = eta2

    param_spiral_inf = param_spiral.copy()
    param_spiral_inf['eta'] = eta3 - theta.std_dev

    param_spiral_mes = param_spiral.copy()
    param_spiral_mes['eta'] = eta3

    param_spiral_sup = param_spiral.copy()
    param_spiral_sup['eta'] = eta3 + theta.std_dev

    nt = 100

    flux_model, err_model, spiral_model_l = curvilinear_flux(
        im_dens_sum, nturn, param_spiral, npts=nt)
    flux_model, err_model, spiral_model_r = curvilinear_flux(
        im_dens_sum, nturn, param_spiral2, npts=nt)
    flux_model, err_model, spiral_model_mes = curvilinear_flux(
        im_dens_sum, nturn, param_spiral_mes, npts=nt)
    flux_model, err_model, spiral_model_inf = curvilinear_flux(
        im_dens_sum, nturn, param_spiral_inf, npts=nt)
    flux_model, err_model, spiral_model_sup = curvilinear_flux(
        im_dens_sum, nturn, param_spiral_sup, npts=nt)

    x_spiral_sup = spiral_model_sup[0]
    y_spiral_sup = spiral_model_sup[1]

    x_spiral_inf = spiral_model_inf[0]
    y_spiral_inf = spiral_model_inf[1]

    fl = interp1d(spiral_model_l[1], spiral_model_l[0],
                  bounds_error=False, kind='cubic', fill_value=np.nan)
    fr_inf = interp1d(y_spiral_inf, x_spiral_inf,
                      bounds_error=False, kind='cubic', fill_value=np.nan)
    fr_sup = interp1d(y_spiral_sup, x_spiral_sup,
                      bounds_error=False, kind='cubic', fill_value=np.nan)

    pos_cut = -70
    e_cut_inf = (fl(pos_cut) - fr_inf(pos_cut)) * pix_mas
    e_cut_sup = (fl(pos_cut) - fr_sup(pos_cut)) * pix_mas
    arm_cut = np.mean([e_cut_inf, e_cut_sup])
    e_arm_cut = np.std([e_cut_inf, e_cut_sup])

    if verbose:
        print('e_arm = %2.1f +/- %2.1f mas' % (arm_cut, e_arm_cut))

    res = AllMyFields({'theta': theta.nominal_value,
                       'e_theta': theta.std_dev,
                       'alpha': alpha[0],
                       'e_alpha': e_alpha[0],
                       'frac': frac.nominal_value,
                       'e_frac': frac.std_dev,
                       's_arm': arm_cut,
                       'e_s_arm': e_arm_cut,
                       'dust_mass': dust_mass})

    vmax = im_dens_sum.max()
    vmin = vmax / 1e4

    if display:
        plt.figure(figsize=(5.5, 7))
        plt.plot(x_schem, y_schem, color='cadetblue',
                 label=r'Face-on ($\theta$ = %2.1f $\pm$ %2.1f$\degree$)' % (theta.nominal_value, theta.std_dev))
        plt.plot(x_schem, -y_schem, color='cadetblue')

        plt.plot(x_schem, y_schem2, color='cadetblue')
        plt.plot(x_schem, -y_schem2, color='cadetblue')

        plt.plot(x_schem, y_schem3, color='crimson',
                 label=r'Edge-on ($\alpha$ = %2.1f $\pm$ %2.1f$\degree$)' % (alpha, e_alpha))
        plt.plot(x_schem, -y_schem3, color='crimson')

        plt.plot(x_schem, y_schem4, color='crimson')
        plt.plot(x_schem, -y_schem4, color='crimson')

        plt.fill_between(x_schem, y_schem, y_schem2, alpha=.2, color='steelblue',
                         label='Dust (f$_{fill}$ = %2.1f $\pm$ %2.1f%%)' % (frac.nominal_value, frac.std_dev))
        plt.fill_between(x_schem, -y_schem, -y_schem2,
                         alpha=.2, color='steelblue')

        plt.fill_between(x_schem, y_schem3, y_schem4, alpha=.2, color='orange')
        plt.fill_between(x_schem, -y_schem3, -y_schem4,
                         alpha=.2, color='orange')
        plt.legend()
        plt.axis([0, 12, -16, 16])
        plt.tight_layout()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(im_dens, norm=LogNorm(), vmin=vmin, vmax=vmax,
                   cmap='gist_earth')  # , extent = extent)
        plt.plot(spiral_model_l[0], spiral_model_l[1], color='crimson', linestyle='--',
                 alpha=.7, label=r'Fit spiral ($\theta$ = 2$\alpha$ = %2.1f$\degree$)' % open_angle)
        plt.plot(spiral_model_r[0], spiral_model_r[1],
                 color='crimson', linestyle='--', alpha=.7)

        plt.axis(np.array([0, npts, npts, 0]) - 0.5)
        plt.legend()

        plt.subplot(1, 2, 2)
        # , vmin = im_dens_sum.max()/1000., vmax = im_dens_sum.max())#, extent = extent)
        plt.imshow(im_dens_sum, norm=LogNorm(), vmin=vmin,
                   vmax=vmax, cmap='gist_earth')
        plt.plot(spiral_model_l[0], spiral_model_l[1],
                 color='crimson', linestyle='--', alpha=.7)
        plt.plot(spiral_model_r[0], spiral_model_r[1],
                 color='crimson', linestyle='--', alpha=.7)
        plt.plot(x_spiral_sup, y_spiral_sup,
                 color='c', linestyle='--', alpha=.7)
        plt.plot(x_spiral_inf, y_spiral_inf,
                 color='c', linestyle='--', alpha=.7)

        plt.plot([fl(pos_cut), fr_sup(pos_cut),
                  fr_inf(pos_cut)], [pos_cut] * 3, '-go')
        plt.text(1.05 * fr_inf(pos_cut), 1.1 * pos_cut,
                 '%2.1f+/-%2.1f' % (arm_cut, e_arm_cut))
        # plt.fill_between(x_spiral_inf, f_spiral_inf(x_spiral_inf), f_spiral_sup(x_spiral_inf), alpha=.4)
        # plt.plot(X_OB, Y_OB, 'c*')
        # plt.plot(X_WR, Y_WR, 'w*')
        # plt.axis(np.array([0, npts, npts, 0])-0.5)
        cb = plt.colorbar()
        cb.set_label(r'$\rho_{dust}$ [g/cm$^2$]')
        plt.subplots_adjust(top=0.968,
                            bottom=0.065,
                            left=0.011,
                            right=0.988,
                            hspace=0.2,
                            wspace=0.002)
        plt.show(block=False)

    return res


def ParamFromHydroFit(N, mix, rnuc, xi, verbose=True):

    # file = open('Saved_param_fitHydro.dpy', 'r+')
    # param = pickle.load(file)
    # file.close()

    param = load_pickle('Saved_param_fitHydro.dpy')

    if mix < param.alpha1.lim_regim:
        a = ufloat(param.alpha1.a, param.alpha1.e_a)
        b = ufloat(param.alpha1.b, param.alpha1.e_b)
        alpha = a * mix + b
    else:
        a = ufloat(param.alpha2.a, param.alpha2.e_a)
        b = ufloat(param.alpha2.b, param.alpha2.e_b)
        alpha = a * mix + b

    if mix < param.omega1.lim_regim:
        a = ufloat(param.omega1.a, param.omega1.e_a)
        b = ufloat(param.omega1.b, param.omega1.e_b)
        omega = a * mix + b
    else:
        a = ufloat(param.omega2.a, param.omega2.e_a)
        b = ufloat(param.omega2.b, param.omega2.e_b)
        omega = a * mix + b

    if mix < param.e_arm1.lim_regim:
        a = ufloat(param.e_arm1.a, param.e_arm1.e_a)
        b = ufloat(param.e_arm1.b, param.e_arm1.e_b)
        e_arm = a * mix + b
    else:
        a = ufloat(param.e_arm2.a, param.e_arm2.e_a)
        b = ufloat(param.e_arm2.b, param.e_arm2.e_b)
        e_arm = a * mix + b

    a = ufloat(param.Mdust.a, param.Mdust.e_a)
    b = ufloat(param.Mdust.b, param.Mdust.e_b)
    c = ufloat(param.Mdust.c, param.Mdust.e_c)
    d = ufloat(param.Mdust.d, param.Mdust.e_d)

    M = 1e-8 * N * xi * (a * rnuc * np.log(mix) + b *
                         np.log(mix) + c * rnuc + d)

    if verbose:
        print('------------------------')
        print('Extrapolated parameters:')
        print('------------------------')
        print('alpha = %2.1f +/- %2.1f deg' %
              (alpha.nominal_value, alpha.std_dev))
        print('omega = %2.1f +/- %2.1f %%' %
              (omega.nominal_value, omega.std_dev))
        print('e_arm = %2.1f +/- %2.1f mas' %
              (e_arm.nominal_value, e_arm.std_dev))
        print('Mdust = %2.1f +/- %2.1f 1e-8 Msun' %
              (M.nominal_value * 1e8, M.std_dev * 1e8))

    return alpha, omega, e_arm, M
