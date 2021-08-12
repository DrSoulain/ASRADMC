# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:31:43 2019

@author: Anthony Soulain (University of Sydney)

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

Functions to analyse and use the saved results of radmc-3d.

------------------------------------------------------------------------
"""

import os
import pickle
import sys

import numpy as np
from astools.all import AllMyFields
from astropy import constants as const
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from scipy import ndimage
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from uncertainties import ufloat
from uncertainties.umath import atan
from termcolor import cprint
ms = const.M_sun.cgs.value  # Solar mass          [g]
au = const.au.cgs.value  # Astronomical Unit       [cm]
m_earth = const.M_earth.cgs.value

flatui = ["#4698cb", "#3fc1c9", "#364f6b", "#fc5185", "#9e978e", "#2ecc71"]

if sys.version[0] == "2":
    from asradmc.ramses import _compute_star_position, ramses_gas_to_dust

fs = 18


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    return pickle_data


def model_linear(x, a, b):
    y = a * x + b
    return y


def model_power(x, a, b, c, d):
    y = a * x ** 3 + b * x ** 2 + c * x + d
    return y


def extract_prf(im,
                pos,
                num=1000,
                display=False,
                p=0.2,
                log=True,
                kernel=1,
                vmin=2e-5,
                vmax=0.2,
                origin=[0, 0],
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

    x_mid = x_min + dx / 2.0
    y_mid = y_min + dy / 2.0

    x_WR = origin[0]
    y_WR = origin[1]

    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    pix = 22.6 / n

    image = np.rot90(im)
    zi = image[x.astype(np.int), y.astype(np.int)]

    xi = np.linspace(0, d, num) * pix

    cond = zi >= 1e-50

    X, Y = xi[cond], medfilt(zi[cond], kernel)

    rnuc = np.sqrt((x_WR - x_mid) ** 2 + (y_WR - y_mid) ** 2)
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

    x_mid2 = x_min + dx / 2.0
    y_mid2 = y_min + dy / 2.0

    rnuc2 = np.sqrt((x_WR - x_mid2) ** 2 + (y_WR - y_mid2) ** 2)
    rnuc_mas2 = rnuc2 * pixel

    w1 = ((m_right[1] - m_left[1]) ** 2 + (m_right[0] - m_left[0]) ** 2) ** 0.5
    w1_pix = w1 * pixel

    alpha_lin = 2 * np.rad2deg(np.arctan(w1_pix / (2 * rnuc_mas2)))

    vgrad = np.gradient(np.rot90(im))
    mag = np.sqrt(vgrad[0] ** 2 + vgrad[1] ** 2)
    lap_grad = ndimage.laplace(mag, output=None, mode="reflect", cval=1)
    lap = ndimage.laplace(im, output=None, mode="reflect", cval=0)

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

    x_mid2 = x_min + dx / 2.0
    y_mid2 = y_min + dy / 2.0

    rnuc3 = np.sqrt((x_WR - x_mid2) ** 2 + (y_WR - y_mid2) ** 2)
    rnuc_mas3 = rnuc3 * pixel

    w1_3 = ((m_right2[1] - m_left2[1]) ** 2 +
            (m_right2[0] - m_left2[0]) ** 2) ** 0.5
    w1_pix3 = w1_3 * pixel

    alpha_lin3 = 2 * np.rad2deg(np.arctan(w1_pix3 / (2 * rnuc_mas3)))

    corr_lap = (lap_grad) - lap

    corr_lap[lap > 0] = 0
    corr_lap[corr_lap < 0] = 0

    if display:
        plt.figure()
        plt.imshow(
            image,
            origin="upper",
            norm=PowerNorm(p),
            cmap="afmhot",
            vmax=vmax,
            vmin=vmin,
        )
        plt.plot([y0, y1], [x0, x1], "w--", lw=1)
        plt.plot(y1, x1, "ro")
        plt.plot(y0, x0, "co")
        # plt.plot(x_mid, y_mid, 'ro')
        plt.plot(x_mid2, y_mid2, "go")
        plt.plot(x_WR, y_WR, "w*")
        plt.plot(m_left[1], m_left[0], "r+")
        plt.plot(m_right[1], m_right[0], "r+")
        plt.plot(m_right2[1], m_right2[0], "b+")
        plt.plot(m_left2[1], m_left2[0], "b+")

        plt.figure()
        plt.plot(X, Y)
        if log:
            plt.yscale("log")

    return X, Y, rnuc_mas, alpha_lin3, alpha_lin


def extract_edge(im, mix, fov=21.6, side=1, orient="h", display=False):
    npts = len(im)
    xy = np.meshgrid(np.arange(npts), np.arange(npts))[0]
    xy2 = np.meshgrid(np.arange(npts), np.arange(npts))[1]

    if orient == "h":
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
    elif mix < 1:
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
        plt.title(r"Edge detection: $\xi$  = %2.2f%%" % mix)
        plt.imshow(lap_left + lap_right)
        plt.plot(edge_right[1], edge_right[0],
                 color="orange", marker="+", ls="")
        plt.plot(edge_left[1], edge_left[0], color="orange", marker="+", ls="")

    return im_left, im_right, edge_left, edge_right


def Open_cube(modeldir, xi, rnuc, mix, amr=True, npix=256):

    if amr:
        s_amr = "Amr"
    else:
        s_amr = "NoAmr"

    imdir = modeldir + \
        "Saved_images/Model_fitSED/rnuc=%2.2f/xi=%2.2f/" % (rnuc, xi)

    imfile = (
        imdir
        + "ramses_cube_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_incl=0_posang=0.00_%s"
        % (npix, xi, rnuc, mix, s_amr)
    )

    dic = load_pickle(imfile + ".dpy")

    data = dic["cube"]
    wl = dic["wl"]
    flux = dic["flux"]
    extent = dic["extent"]

    band = {
        "0.55": "V",
        "1.21": "J",
        "1.65": "H",
        "2.27": "K",
        "3.5": "L",
        "10.0": "N",
        "300.0": "band3",
    }

    cube = {}
    for i in range(len(data)):
        cube[band[str(wl[i])]] = {"im": data[i], "wl": wl[i], "f": flux[i]}
    cube["extent"] = extent
    cube["xi"] = xi
    cube["mix"] = mix
    cube["rnuc"] = rnuc

    return AllMyFields(cube)


def norm(tab):
    return tab / np.max(tab)


def spiral_archimedian(t, param):
    a = param["d_arm"] / (2 * np.pi) / param["pixel_size"]
    eta = np.deg2rad(param["eta"])  # angle sky
    theta = np.deg2rad(param["theta"])  # angle incl x
    phi = np.deg2rad(param["phi"])  #
    x = (a * t) * (np.cos(t + eta))
    y = (a * t) * (np.sin(t + eta))

    # rotation inclinaison
    xp = x
    yp = y * np.cos(theta)

    # Rotation projection
    xpp = xp * np.cos(phi) - yp * np.sin(phi) + param["x0"]
    ypp = xp * np.sin(phi) + yp * np.cos(phi) + param["y0"]
    return xpp, ypp


def curvilinear_flux(image, turn, param, npts=200):
    """
    Compute curvilinear flux along the archimedian spiral.
    """
    t = np.linspace(0, turn * 2 * np.pi, npts)
    ui, vi = spiral_archimedian(t, param)
    flux = []
    x = np.arange(len(image))
    y = np.arange(len(image))
    f = interp2d(x, y, image, "cubic")
    for xx in np.arange(len(ui)):
        flux.append(f(ui[xx], vi[xx])[0])

    flux = np.array(flux)

    flux[flux < 0] = 1e-10

    err_flux_rel = np.sqrt(flux) / flux

    flux_norm = 100 * norm(np.array(flux))

    err_flux = err_flux_rel * flux_norm
    return flux_norm, err_flux, [ui, vi]


def compute_omega(rho_dust, dust_param, dict_simu,
                  n_prf=10, f0=2, rmax=14, details=False,
                  save=False, verbose=True, display=False):
    """ Compute the internal filling factor of the spiral using 
    circle profils at different radius (centered on x0, y0). The
    list of radius is computed as np.linspace(f0 * dust_param['rnuc'] * 1e3, rmax, 
    n_prf).

    Parameters:
    -----------
    `rho_dust` {array}:
        3d dust density cube from ramses.ramses_gas_to_dust(),\n
    `dust_param` {dict}:
        Parameters of the dust models,\n
    `dict_simu` {dict}:
        Dictionnary of the ramses simulation ramses.amr_to_cube(),\n
    `n_prf` {int}:
        Number of extracted profils,\n
    `f0` (float):
        Fraction of rnuc where the first profils is extracted,\n
    `rmax` {float}:
        Last radius to extract profils,\n
    `details` (bool):
        If True, print the omega for each radii.

    Outputs:
    --------
    `frac` {ufloat}:
        Filling factor omega averaged over several radius [%],\n
    `theta` {ufloat}:
        Opening angle-like computed as angular thickness using the 
        edges of each dusty profils [deg].
    """

    npts = len(rho_dust)
    i_t = npts // 2
    image = np.rot90(rho_dust[:, :, i_t])
    im_dens_sum = np.rot90(rho_dust.sum(axis=2))

    cond_extrapol = im_dens_sum < 1e-19

    im_dens_sum[cond_extrapol] = 1e-50
    image[cond_extrapol] = 1e-50

    p_star = _compute_star_position(dict_simu)
    pix_mas = p_star.pix_mas,
    x0, y0 = p_star.x_OB, p_star.y_OB

    tmin = -0.5
    tmax = np.pi / 1.2

    t2 = np.linspace(tmax, tmin, 300)

    l_frac, l_theta = [], []

    l_r = np.linspace(f0 * dust_param['rnuc'] * 1e3, rmax,
                      n_prf)

    if details:
        print('--------------------------------------')
        print('Filling factor at different radii:')

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

        if details:
            print(r'r = %2.1f mas, theta = %2.1f deg, filling factor = %2.1f %%' % (
                r, t_thickness, frac_dust))

        if save:
            savedict = {'t2': t2, 'flux': flux, 'r': r, 'l_r': l_r,
                        'cond_large': cond_large, 'cond_dust': cond_dust,
                        't_l': t_l, 't_r': t_r, 'f_l': f_l,
                        'f_r': f_r, 'frac_dust': frac_dust}
            file = open('saveprf/saveprf%2.1f.dpy' % r, 'wb')
            pickle.dump(savedict, file, 2)
            file.close()

        if display:
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
            plt.plot(x0, y0, 'r+', label='OB Star')
            plt.legend(fontsize=10, loc=2)

            plt.axis(np.array([0, npts, npts, 0]) - 0.5)
            plt.subplots_adjust(top=0.968,
                                bottom=0.065,
                                left=0.05,
                                right=0.988,
                                hspace=0.2,
                                wspace=0.002)
    if details:
        print('--------------------------------------')
    l_frac, l_theta = np.array(l_frac), np.array(l_theta)
    omega = ufloat(np.mean(l_frac), np.std(l_frac) / np.sqrt(len(l_r)))
    theta = ufloat(np.mean(l_theta), np.std(l_theta) / np.sqrt(len(l_r)))

    if verbose:
        print('omega = %2.1f +/- %2.1f %%' % (omega.nominal_value,
                                              omega.std_dev))
        print('theta_in = %2.1f +/- %2.1f deg' % (theta.nominal_value,
                                                  theta.std_dev))
    return omega, theta


def compute_theta(rho_dust, dust_param, dict_simu,
                  n_prf=10, f0=2, rmax=14, details=False,
                  save=False, verbose=True, display=False):
    """ Compute the internal filling factor of the spiral using 
    circle profils at different radius (centered on x0, y0). The
    list of radius is computed as np.linspace(f0 * dust_param['rnuc'] * 1e3, rmax, 
    n_prf).

    Parameters:
    -----------
    `rho_dust` {array}:
        3d dust density cube from ramses.ramses_gas_to_dust(),\n
    `dust_param` {dict}:
        Parameters of the dust models,\n
    `dict_simu` {dict}:
        Dictionnary of the ramses simulation ramses.amr_to_cube(),\n
    `n_prf` {int}:
        Number of extracted profils,\n
    `f0` (float):
        Fraction of rnuc where the first profils is extracted,\n
    `rmax` {float}:
        Last radius to extract profils,\n
    `details` (bool):
        If True, print the omega for each radii.

    Outputs:
    --------
    `frac` {ufloat}:
        Filling factor omega averaged over several radius [%],\n
    `theta` {ufloat}:
        Opening angle-like computed as angular thickness using the 
        edges of each dusty profils [deg].
    """

    npts = len(rho_dust)
    i_t = npts // 2
    image = np.rot90(rho_dust[:, :, i_t])
    im_dens_sum = np.rot90(rho_dust.sum(axis=2))

    cond_extrapol = im_dens_sum < 1e-20

    im_dens_sum[cond_extrapol] = 1e-50
    image[cond_extrapol] = 1e-50

    p_star = _compute_star_position(dict_simu)
    pix_mas = p_star.pix_mas,
    x0, y0 = p_star.x_OB, p_star.y_OB

    tmin = -0.5
    tmax = np.pi / 1.2

    t2 = np.linspace(tmax, tmin, 300)

    l_theta = []

    l_r = np.linspace(f0 * dust_param['rnuc'] * 1e3, rmax,
                      n_prf)

    if details:
        print('--------------------------------------')
        print('Filling factor at different radii:')
    for r in l_r:
        x_ring = r * np.cos(t2) / pix_mas + x0  # + 20
        y_ring = r * np.sin(t2) / pix_mas + y0

        x = np.arange(npts)
        y = np.arange(npts)
        f = interp2d(x, y, im_dens_sum, 'cubic', copy=False,
                     bounds_error=False, fill_value=0)
        flux = []
        for xx in xrange(len(x_ring)):
            flux.append(f(x_ring[xx], y_ring[xx])[0])
        flux = np.array(flux)
        flux[flux < 1e-20] = 1e-20

        cond_dust = flux >= 1e-17

        t_l = t2[cond_dust][0]
        t_r = t2[cond_dust][-1]
        f_l = flux[cond_dust][0]
        f_r = flux[cond_dust][-1]

        cond_large = (t2 <= t_l) & (t2 >= t_r)

        n_thickness = float(len(flux[cond_large]))

        pix_t = abs(np.diff(t2)[0])
        t_thickness = round(np.rad2deg(n_thickness * pix_t), 2)

        l_theta.append(t_thickness)

        if details:
            print(r'r = %2.1f mas, theta = %2.1f deg' % (
                r, t_thickness))

        if save:
            savedict = {'t2': t2, 'flux': flux,
                        'cond_large': cond_large, 'cond_dust': cond_dust,
                        't_l': t_l, 't_r': t_r, 'f_l': f_l,
                        'f_r': f_r}
            file = open('saveprf%s.dpy' % r, 'wb')
            pickle.dump(savedict, file, 2)
            file.close()

        if display:
            vmax = image.max()
            vmin = vmax / 1e4
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.semilogy(t2, flux, color='lightgrey', lw=1,
                         label='Interpolated density')
            plt.semilogy(t2[cond_large], flux[cond_large], 'o',
                         color='cadetblue', label='Inside spiral')
            plt.semilogy([t_l, t_r], [f_l, f_r], 's', color='crimson',
                         label=r'Thickness (%2.1f$\degree$)' % t_thickness)
            plt.legend(fontsize=10, loc=3)
            plt.xlim(tmax, tmin)
            plt.subplot(1, 2, 2)
            plt.imshow(im_dens_sum, norm=LogNorm(),
                       cmap='gist_earth', vmin=vmin, vmax=vmax)
            plt.plot(x_ring, y_ring, color='lightyellow',
                     label='Extracted profil (r = %2.1f mas)' % (r))
            plt.plot(x0, y0, 'r+', label='OB Star')
            plt.legend(fontsize=10, loc=2)

            plt.axis(np.array([0, npts, npts, 0]) - 0.5)
            plt.subplots_adjust(top=0.968,
                                bottom=0.065,
                                left=0.05,
                                right=0.988,
                                hspace=0.2,
                                wspace=0.002)
    if details:
        print('--------------------------------------')

    l_theta = np.array(l_theta)
    theta = ufloat(np.mean(l_theta), np.std(l_theta) / np.sqrt(len(l_r)))

    if verbose:
        print('theta = %2.1f +/- %2.1f deg' % (theta.nominal_value,
                                               theta.std_dev))
    return theta


def explor_Tsub(npix, rnuc, xi, mix, modeldir, amr=True, Tsub=2000, Tm=3500, Tstep=100,
                display=False):
    """ 

    Explore the pre-computed temperature distribution grid from radmc3d. 

    Returns
    -------
    `n_over_Tsub` {int}: 
        N cells over Tsub,\n
    `n_over_rel` {float}: 
        Relative N cells hot [%],\n
    `m_hot_dust` {float}: 
        hot dust in mass of pluto (0.0022 M_earth, 1.3e22 kg),\n
    `Tmax` {float}:
        Maximum temperature found.
    """

    if amr:
        s_amr = "AMR"
    else:
        s_amr = "NoAmr"

    Tfile = modeldir + "Saved_temp/"
    Tfile += 'rnuc=%2.2f/xi=%2.2f/' % (rnuc, xi)
    Tfile += "Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s" % (
        npix, xi, rnuc, mix, s_amr)

    Rhofile = (modeldir + "Saved_dens/Rhodist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s" %
               (npix, xi, rnuc, mix, s_amr))

    dic_T = load_pickle(Tfile + ".dpy")

    hdu = fits.open(Tfile + ".fits")
    Tdist = hdu[0].data

    cond = Tdist > 0

    try:
        dic_rho = load_pickle(Rhofile + ".dpy")
        rho = dic_rho["rho"]
        Vol_cell = dic_rho["Vol_cell"]
        mdust_tot = dic_rho["mass_tot"]
        cond_sub = Tdist >= Tsub
        rho_hot = rho[cond_sub]
        V_hot = Vol_cell[cond_sub]
        m_dust_hot = (rho_hot * V_hot).sum() / ms
        frac_hot_dust = 100 * m_dust_hot / mdust_tot
    except Exception:
        frac_hot_dust = np.nan

    n_over_Tsub = dic_T["Nhot"]
    n_dust = float(len(Tdist[cond]))
    n_over_rel = 100 * (n_over_Tsub / n_dust)
    Tmean = dic_T["Tmean"]
    Tmax = dic_T["Tmax"]

    if np.isnan(frac_hot_dust):
        m_hot_dust = dic_T['param']['dustmass'] * \
            n_over_rel / 100. * (1. / (m_earth / ms)) / 0.00218

    if display:
        plt.figure()
        plt.hist(Tdist[cond & (Tdist < Tsub)], bins=np.arange(0, Tm, Tstep))
        plt.hist(
            Tdist[Tdist >= Tsub],
            bins=np.arange(0, Tm, Tstep),
            color="crimson",
            alpha=0.5,
            label="Hot cells = %i (%2.3f%%)"
            % (n_over_Tsub, 100 * (n_over_Tsub / n_dust)),
        )

        plt.vlines(Tmean, 1e-1, npix ** 3,
                   label=r"T$_{mean}$ (%2.0f K)" % Tmean)
        plt.vlines(
            Tsub,
            1e-1,
            npix ** 3,
            linestyle="--",
            color="r",
            label=r"T$_{sub}$ (%2.0f K)" % Tsub,
        )
        plt.legend(loc="best", fontsize=8)
        plt.xlabel("T [K]", fontsize=12, color="gray")
        plt.ylabel("# cells", fontsize=12, color="gray")
        plt.yscale("log")
        plt.ylim(1, 1e7)
        plt.xlim(0, 3500)
        plt.tight_layout()

    return n_over_Tsub, n_over_rel, m_hot_dust, Tmax


def ParamFromHydroFit(N, mix, rnuc, xi, verbose=True,
                      file=None, version=2):
    try:
        param = load_pickle("./Saved_param_fitHydro_v%i.dpy" % version)
    except IOError:
        if file is not None:
            try:
                param = load_pickle(file)
            except IOError:
                cprint('\nResults dictionnary not found (%s).' % file, 'red')
        else:
            cprint('\nResults dictionnary not found, give file to use instead.',
                   'red')
            return None

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

    rnuc_au = rnuc * 2.58
    ar = 3e-7 * (a * rnuc_au + b)
    br = 3e-7 * (c * rnuc_au + d)

    M = N * xi * (ar * np.log(mix) + br)

    if verbose:
        print("------------------------")
        print("Extrapolated parameters:")
        print("------------------------")
        print("alpha = %2.1f +/- %2.1f deg" %
              (alpha.nominal_value, alpha.std_dev))
        print("omega = %2.1f +/- %2.1f %%" %
              (omega.nominal_value, omega.std_dev))
        print("e_arm = %2.1f +/- %2.1f AU" %
              (e_arm.nominal_value, e_arm.std_dev))
        print(
            "Mdust = %2.1f +/- %2.1f 1e-7 Msun"
            % (M.nominal_value * 1e7, M.std_dev * 1e7)
        )

    return alpha, omega, e_arm, M


def compute_alpha(rho_dust, mix, l_max=20, expert_plot=False,
                  verbose=True, display=True):
    """ Compute the opening angle from the good size of the simulation. """
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
        lim_r = 2e-16
    else:
        dr = 65
        lim_r = .6e-16

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

        x_mod = np.linspace(-150, 350, 100)
        y_mod_u = model_linear(x_mod, *popt_u)
        y_mod_d = model_linear(x_mod, *popt_d)
    except TypeError:
        a_deg = ufloat(0, 0)
        x_mod = y_mod_u = y_mod_d = None

    if verbose:
        print('alpha = %2.1f +/- %2.1f deg' % (a_deg.nominal_value,
                                               a_deg.std_dev))
    if display:
        plt.figure()
        plt.clf()
        plt.title(r'Edge detection: $\xi$  = %2.2f%%' % mix)
        plt.imshow(im, cmap='gist_earth')
        plt.plot(xl, yl, 'w+')
        plt.plot(xr, yr, 'w+')
        try:
            plt.plot(x_mod, y_mod_u, '--', color='crimson')
            plt.plot(x_mod, y_mod_d, '--', color='crimson')
        except Exception:
            pass
        plt.axis([0, len(im), len(im), 0])
        plt.tight_layout()

        lap_left[lap_left < 0] = 0
        lap_right[lap_right < 0] = 0

        if expert_plot:
            plt.figure()
            plt.title(r'Edge detection: $\xi$  = %2.2f%%' % mix)
            plt.imshow(lap_left + lap_right, cmap='gist_earth')
            plt.colorbar()
            plt.plot(xr, yr,
                     color='orange', marker='+', ls='')
            plt.plot(xl, yl,
                     color='orange', marker='+', ls='')
    return a_deg, im, edge_left, edge_right, x_mod, y_mod_u, y_mod_d


def _plot_edges(e_u, e_d, pix_size, npts):
    x_edge_u, y_edge_u = pix_size * \
        (e_u[1] - npts // 2), pix_size * (e_u[0] - npts // 2)
    x_edge_d, y_edge_d = pix_size * \
        (e_d[1] - npts // 2), pix_size * (e_d[0] - npts // 2)

    plt.plot(x_edge_u, y_edge_u, '+', color='w',
             ms=10)
    plt.plot(x_edge_d, y_edge_d, '+', color='w', ms=10)


def Compute_alpha_vs_mix(l_mix, dust_param, dict_simu, fov=0.0216, display=False, save=False):
    x_mod = np.linspace(-150, 250, 100)

    l_alpha_side, l_alpha_e = [], []

    j = 0
    for i_m in l_mix:
        dust_param['mix'] = i_m / 100.
        rho_dust, gridinfo, dust_mass = ramses_gas_to_dust(
            dict_simu, dust_param, display=False)

        alpha0, im0, e_u2, e_d2, x_mod2, y_mod_u2, y_mod_d2 = compute_alpha(
            rho_dust, i_m, display=False)

        im_dens_sum = np.rot90(rho_dust.sum(axis=1))

        A, B, e_u, e_d = extract_edge(im_dens_sum, i_m,
                                      orient='h', display=False)

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

        # x_edge_u, y_edge_u = pix_size * \
        #     (e_u[1] - npts // 2), pix_size * (e_u[0] - npts // 2)
        # x_edge_d, y_edge_d = pix_size * \
        #     (e_d[1] - npts // 2), pix_size * (e_d[0] - npts // 2)
        x_mod_c = pix_size * (x_mod - npts // 2)
        y_mod_u_c = pix_size * (y_mod_u - npts // 2)
        y_mod_d_c = pix_size * (y_mod_d - npts // 2)

        try:
            x_mod_c2 = pix_size * (x_mod2 - npts // 2)
            y_mod_u_c2 = pix_size * (y_mod_u2 - npts // 2)
            y_mod_d_c2 = pix_size * (y_mod_d2 - npts // 2)
        except TypeError:
            pass

        if display:
            plt.figure(figsize=(9, 4.5))
            plt.subplot(1, 2, 1)
            plt.imshow(im_dens_sum, cmap='gist_earth', extent=extent,
                       origin='lower')
            plt.xlabel('X [AU]', fontsize=fs)
            plt.ylabel('Y [AU]', fontsize=fs)
            _plot_edges(e_u, e_d, pix_size, npts)
            plt.plot(x_mod_c, y_mod_u_c, '--', lw=3,
                     color='crimson', label=r'$\alpha_s$ = %2.1f deg' % (a_deg.nominal_value))
            plt.plot(x_mod_c, y_mod_d_c, '--', lw=2, color='crimson')
            plt.tick_params(left='off', bottom='off')
            plt.legend()
            plt.ylim(extent[0], extent[1])
            plt.xlim(extent[0], extent[1])

            plt.subplot(1, 2, 2)
            plt.imshow(im0, cmap='gist_earth', extent=extent,
                       origin='lower')
            _plot_edges(e_u2, e_d2, pix_size, npts)
            try:
                plt.plot(x_mod_c2, y_mod_u_c2, '--', lw=3,
                         color='crimson', label=r'$\alpha$ = %2.1f deg' % (alpha0.nominal_value))
                plt.plot(x_mod_c2, y_mod_d_c2, '--', lw=3, color='crimson')
            except Exception:
                pass
            plt.legend()
            plt.ylim(extent[0], extent[1])
            plt.xlim(extent[0], extent[1])
            plt.xlabel('X [AU]', fontsize=fs)
            plt.tight_layout()
            plt.subplots_adjust(right=1, left=0.08, bottom=0.11, top=1,
                                wspace=0.11)

            if save:
                if j <= 9:
                    plt.savefig('anim_alpha_mix/im00%i.png' %
                                j)  # , dpi = 300)
                elif j <= 99:
                    plt.savefig('anim_alpha_mix/im0%i.png' % j)  # , dpi = 300)
                else:
                    plt.savefig('anim_alpha_mix/im%i.png' % j)  # , dpi = 300)
            j += 1

    # if save:
    #     os.system('rm anim_alpha_mix/alpha_vs_mix.gif')
    #     os.system('rm anim_alpha_mix/alpha_vs_mix.mp4')
    #     (
    #         ffmpeg.input('anim_alpha_mix/im*.png',
    #                      pattern_type='glob', framerate=10)
    #         .output('anim_alpha_mix/alpha_vs_mix.gif')
    #         .run()
    #     )
    #     (
    #         ffmpeg.input('anim_alpha_mix/*.png',
    #                      pattern_type='glob', framerate=15)
    #         .output('anim_alpha_mix/alpha_vs_mix.mp4')
    #         .run()
    #     )

    l_alpha = np.array(l_alpha_side)
    l_alpha_e = np.array(l_alpha_e)
    l_alpha = np.array([alpha0.nominal_value])
    l_alpha_e = np.array([alpha0.std_dev])

    return rho_dust, gridinfo, dust_mass, l_alpha, l_alpha_e


def check_model_done(gs, npix=256, noutput=270, nphot=1e8, amr=True):
    s_amr = 'NoAmr'
    if amr:
        s_amr = 'AMR'

    modeldir = '/Volumes/SpaceStone/RADMC3D/PostRamses_Jan2021/'
    modeldir1 = modeldir + 'ResultWR104_%i_npix=%i_gs=%s_nphot=%2.1e/' % (
        noutput, npix, gs, nphot)
    modeldir2 = modeldir + 'ResultWR104_%i_npix=%i_gs=%s_nphot=%2.1e/' % (
        noutput, npix // 2, gs, nphot)

    list_mix = [1, 5, 10, 20]
    list_rnuc = [15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35]
    list_xi = [0.1, 1, 3, 5, 7, 10]

    mat2 = np.zeros([len(list_rnuc), len(list_mix), len(list_xi)])
    list_xi2 = list_xi[::-1]
    for i in range(mat2.shape[0]):
        rnuc = list_rnuc[i]
        for j in range(mat2.shape[1]):
            mix = list_mix[j]
            for k in range(mat2.shape[2]):
                xi = list_xi2[k]
                Tfile = modeldir1 + "Saved_temp/"
                Tfile += 'rnuc=%2.2f/xi=%2.2f/' % (rnuc, xi)
                Tfile += "Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s.fits" % (
                    npix, xi, rnuc, mix, s_amr)
                if os.path.exists(Tfile):
                    mat2[i, j, k] = 1
                else:
                    Tfile2 = modeldir2 + "Saved_temp/"
                    Tfile2 += 'rnuc=%2.2f/xi=%2.2f/' % (rnuc, xi)
                    Tfile2 += "Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s.fits" % (
                        npix // 2, xi, rnuc, mix, s_amr)
                    if os.path.exists(Tfile2):
                        mat2[i, j, k] = 2

    dic_color = {'1': 'g', '0': 'r', '2': 'gold'}
    dic_marker = {'1': 'o', '0': 'x', '2': 'o'}

    fig = plt.figure(figsize=(8, 3))
    plt.title(modeldir1.split('/')[-2])
    ncol = 0
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            mix = list_mix[j]
            for k in range(mat2.shape[2]):
                xi = list_xi[k]
                itexist = str(int(mat2[i, j, k]))
                plt.plot(j + 6 * ncol, k, color=dic_color[itexist],
                         marker=dic_marker[itexist])
        ncol += 1
    for k in range(mat2.shape[2]):
        xi = list_xi2[k]
        plt.text(-5, k, r'$\xi$ = %2.1f%%' % (xi), fontsize=8,
                 ha='center', va='center')

    for i in range(mat2.shape[0]):
        rnuc = list_rnuc[i]
        plt.text(1.5 + (6 * i), 5.5, '%2.1f' % (rnuc), fontsize=8,
                 ha='center', va='center', color='blue')
    plt.text(-5, 5.5, 'rnuc [mas]', fontsize=8,
             ha='center', va='center', color='blue')
    plt.plot(np.nan, 0, 'rx', label='Not computed')
    plt.plot(np.nan, 0, 'go', label='Done')
    plt.plot(np.nan, 0, 'o', color='gold', label='Done (npix//2)')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    plt.axis([-10, 54, -0.5, 6])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return fig
# def Calc_param_hydro(mix, dict_simu, dust_param, dpc=2580, nturn=0.6,
#                      step=0.066, eta1=180, save=False, verbose=True, display=True):

#     npts = dict_simu['mix'].shape[0]

#     fov = dict_simu['info']['fov'] / 1000.  # [arcsec]

#     # pix_cm = fov * dpc * au / npts
#     # pix_au = fov * dpc / npts
#     pix_mas = (fov * 1000.) / npts

#     unit = 'pix'

#     if unit == 'mas':
#         extent = np.array([-fov / 2., fov / 2, -fov / 2, fov / 2]) * 1000
#         pix_size = (2 * abs(extent[0])) / npts
#     elif unit == 'au':
#         extent = (np.array([-fov / 2., fov / 2, -fov / 2, fov / 2])) * dpc
#         pix_size = (2 * abs(extent[0])) / npts
#     elif unit == 'cm':
#         extent = (np.array([-fov / 2., fov / 2, -fov / 2, fov / 2])) * dpc * au
#         pix_size = (2 * abs(extent[0])) / npts
#     else:
#         extent = np.array([-npts / 2., npts / 2, -npts / 2, npts / 2])
#         pix_size = (2 * abs(extent[0])) / npts

#     X_WR = dict_simu['info']['p_WR'][0] - 0.5
#     Y_WR = npts - dict_simu['info']['p_WR'][1] - 0.5

#     sep = dict_simu['info']['sep']
#     bin_phase = dict_simu['info']['bin_phase']

#     x_OB = sep * np.cos(np.deg2rad(-bin_phase))
#     y_OB = sep * np.sin(np.deg2rad(-bin_phase))

#     X_OB = X_WR + x_OB / ((fov * 1000.) / npts) + 1
#     Y_OB = Y_WR - y_OB / ((fov * 1000.) / npts)

#     X_OB, Y_OB = X_OB * pix_size, Y_OB * pix_size
#     X_WR, Y_WR = X_WR * pix_size, Y_WR * pix_size

#     rho_dust, gridinfo, dust_mass, alpha, e_alpha = Compute_alpha_vs_mix(
#         [mix], dust_param, dict_simu, display=display)

#     i_t = npts // 2
#     # im_gaz = np.rot90(gridinfo.gaz_grid[:,:,i_t])
#     # im_mix = np.rot90(gridinfo.mix[:,:,i_t])
#     im_dens = np.rot90(rho_dust[:, :, i_t])
#     im_dens_sum = np.rot90(rho_dust.sum(axis=2))

#     cond_extrapol = im_dens_sum < 1e-19

#     im_dens_sum[cond_extrapol] = 1e-50
#     im_dens[cond_extrapol] = 1e-50

#     l_r = np.linspace(2 * dust_param['rnuc'] * 1e3, 14, 8)

#     l_frac, l_theta = circle_profil_r(l_r, im_dens, pix_mas,
#                                       X_OB, Y_OB, verbose=verbose,
#                                       display=display)

#     frac = ufloat(np.mean(l_frac), np.std(l_frac))
#     theta = ufloat(np.mean(l_theta), np.std(l_theta))
#     # theta_rad = theta * np.pi/180.

#     # e_thick = 2*rnuc*tan(theta_rad/2.)

#     if verbose:
#         print('\nFilling_factor = %2.1f +/- %2.1f %%' %
#               (frac.nominal_value, frac.std_dev))
#         print('Theta = %2.1f +/- %2.1f deg' %
#               (theta.nominal_value, theta.std_dev))
#         print('Alpha = %2.1f +/- %2.1f deg' % (alpha, e_alpha))
#         # print('e_arm = %2.1f +/- %2.1f mas'%(e_thick.nominal_value, e_thick.std_dev))

#     angle = theta.nominal_value / 2.
#     angle2 = (1 - (frac.nominal_value / 100.)) * angle
#     angle3 = alpha / 2.
#     angle4 = (1 - (frac.nominal_value / 100.)) * angle3

#     x_schem = np.linspace(0, 300, 10)
#     coef1 = np.sin(np.deg2rad(angle)) / np.cos(np.deg2rad(angle))
#     coef2 = np.sin(np.deg2rad(angle2)) / np.cos(np.deg2rad(angle2))
#     coef3 = np.sin(np.deg2rad(angle3)) / np.cos(np.deg2rad(angle3))
#     coef4 = np.sin(np.deg2rad(angle4)) / np.cos(np.deg2rad(angle4))

#     y_schem = coef1 * x_schem
#     y_schem2 = coef2 * x_schem
#     y_schem3 = coef3 * x_schem
#     y_schem4 = coef4 * x_schem

#     open_angle = alpha  # theta.nominal_value#alpha
#     # eta1 = 180
#     eta2 = eta1 + open_angle
#     eta3 = eta1 + theta.nominal_value

#     param_spiral = {
#         'eta': eta1,  # 181.5            #rotation phase
#         'theta': 180,  # 2d projection angle
#         'phi': 0,  # Inclinaison line of sight
#         'x0': X_OB,
#         'y0': Y_OB,
#         'd_arm': step * 1e3,
#         'pixel_size': fov / npts * 1000.}

#     param_spiral2 = param_spiral.copy()
#     param_spiral2['eta'] = eta2

#     param_spiral_inf = param_spiral.copy()
#     param_spiral_inf['eta'] = eta3 - theta.std_dev

#     param_spiral_mes = param_spiral.copy()
#     param_spiral_mes['eta'] = eta3

#     param_spiral_sup = param_spiral.copy()
#     param_spiral_sup['eta'] = eta3 + theta.std_dev

#     nt = 100

#     flux_model, err_model, spiral_model_l = curvilinear_flux(
#         im_dens_sum, nturn, param_spiral, npts=nt)
#     flux_model, err_model, spiral_model_r = curvilinear_flux(
#         im_dens_sum, nturn, param_spiral2, npts=nt)
#     flux_model, err_model, spiral_model_mes = curvilinear_flux(
#         im_dens_sum, nturn, param_spiral_mes, npts=nt)
#     flux_model, err_model, spiral_model_inf = curvilinear_flux(
#         im_dens_sum, nturn, param_spiral_inf, npts=nt)
#     flux_model, err_model, spiral_model_sup = curvilinear_flux(
#         im_dens_sum, nturn, param_spiral_sup, npts=nt)

#     x_spiral_sup = spiral_model_sup[0]
#     y_spiral_sup = spiral_model_sup[1]

#     x_spiral_inf = spiral_model_inf[0]
#     y_spiral_inf = spiral_model_inf[1]

#     fl = interp1d(spiral_model_l[1], spiral_model_l[0],
#                   bounds_error=False, kind='cubic', fill_value=np.nan)
#     fr_inf = interp1d(y_spiral_inf, x_spiral_inf,
#                       bounds_error=False, kind='cubic', fill_value=np.nan)
#     fr_sup = interp1d(y_spiral_sup, x_spiral_sup,
#                       bounds_error=False, kind='cubic', fill_value=np.nan)

#     pos_cut = -70
#     e_cut_inf = (fl(pos_cut) - fr_inf(pos_cut)) * pix_mas
#     e_cut_sup = (fl(pos_cut) - fr_sup(pos_cut)) * pix_mas
#     arm_cut = np.mean([e_cut_inf, e_cut_sup])
#     e_arm_cut = np.std([e_cut_inf, e_cut_sup])

#     if verbose:
#         print('e_arm = %2.1f +/- %2.1f mas' % (arm_cut, e_arm_cut))

#     res = AllMyFields({'theta': theta.nominal_value,
#                        'e_theta': theta.std_dev,
#                        'alpha': alpha[0],
#                        'e_alpha': e_alpha[0],
#                        'frac': frac.nominal_value,
#                        'e_frac': frac.std_dev,
#                        's_arm': arm_cut,
#                        'e_s_arm': e_arm_cut,
#                        'dust_mass': dust_mass})

#     vmax = im_dens_sum.max()
#     vmin = vmax / 1e4

#     if display:
#         plt.figure(figsize=(5.5, 7))
#         plt.plot(x_schem, y_schem, color='cadetblue',
#                  label=r'Face-on ($\theta$ = %2.1f $\pm$ %2.1f$\degree$)' % (theta.nominal_value, theta.std_dev))
#         plt.plot(x_schem, -y_schem, color='cadetblue')

#         plt.plot(x_schem, y_schem2, color='cadetblue')
#         plt.plot(x_schem, -y_schem2, color='cadetblue')

#         plt.plot(x_schem, y_schem3, color='crimson',
#                  label=r'Edge-on ($\alpha$ = %2.1f $\pm$ %2.1f$\degree$)' % (alpha, e_alpha))
#         plt.plot(x_schem, -y_schem3, color='crimson')

#         plt.plot(x_schem, y_schem4, color='crimson')
#         plt.plot(x_schem, -y_schem4, color='crimson')

#         plt.fill_between(x_schem, y_schem, y_schem2, alpha=.2, color='steelblue',
#                          label='Dust (f$_{fill}$ = %2.1f $\pm$ %2.1f%%)' % (frac.nominal_value, frac.std_dev))
#         plt.fill_between(x_schem, -y_schem, -y_schem2,
#                          alpha=.2, color='steelblue')

#         plt.fill_between(x_schem, y_schem3, y_schem4, alpha=.2, color='orange')
#         plt.fill_between(x_schem, -y_schem3, -y_schem4,
#                          alpha=.2, color='orange')
#         plt.legend()
#         plt.axis([0, 12, -16, 16])
#         plt.tight_layout()

#         plt.figure(figsize=(14, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(im_dens, norm=LogNorm(), vmin=vmin, vmax=vmax,
#                    cmap='gist_earth')  # , extent = extent)
#         plt.plot(spiral_model_l[0], spiral_model_l[1], color='crimson', linestyle='--',
#                  alpha=.7, label=r'Fit spiral ($\alpha$ = %2.1f$\degree$)' % open_angle)
#         plt.plot(spiral_model_r[0], spiral_model_r[1],
#                  color='crimson', linestyle='--', alpha=.7)

#         plt.axis(np.array([0, npts, npts, 0]) - 0.5)
#         plt.legend()

#         plt.subplot(1, 2, 2)

#         plt.imshow(im_dens_sum, norm=LogNorm(), vmin=vmin,
#                    vmax=vmax, cmap='gist_earth')
#         plt.plot(spiral_model_l[0], spiral_model_l[1],
#                  color='crimson', linestyle='--', alpha=.7)
#         plt.plot(spiral_model_r[0], spiral_model_r[1],
#                  color='crimson', linestyle='--', alpha=.7)
#         plt.plot(x_spiral_sup, y_spiral_sup,
#                  color='c', linestyle='--', alpha=.7)
#         plt.plot(x_spiral_inf, y_spiral_inf,
#                  color='c', linestyle='--', alpha=.7)

#         plt.plot([fl(pos_cut), fr_sup(pos_cut),
#                   fr_inf(pos_cut)], [pos_cut] * 3, '-go')
#         plt.text(1.05 * fr_inf(pos_cut), 1.1 * pos_cut,
#                  '%2.1f+/-%2.1f AU' % (arm_cut * 2.58, e_arm_cut * 2.58))
#         # plt.fill_between(x_spiral_inf, f_spiral_inf(x_spiral_inf), f_spiral_sup(x_spiral_inf), alpha=.4)
#         # plt.plot(X_OB, Y_OB, 'c*')
#         # plt.plot(X_WR, Y_WR, 'w*')
#         # plt.axis(np.array([0, npts, npts, 0])-0.5)
#         cb = plt.colorbar()
#         cb.set_label(r'$\rho_{dust}$ [g/cm$^2$]')
#         plt.subplots_adjust(top=0.968,
#                             bottom=0.065,
#                             left=0.011,
#                             right=0.988,
#                             hspace=0.2,
#                             wspace=0.002)
#         plt.show(block=False)

#     return res


def compute_s_arm(alpha, eta1=0, step=0.066, dpc=2.58,
                  turn=0.66, npts=1000, pos_cut=10, display=True,
                  verbose=True):
    """ Compute arm thickness [AU] for a given opening angle alpha (need
    step [mas], distance dpc [kpc], turn and a Y position to extract 
    the thickness). """
    param1 = {'eta': eta1,  # 181.5            #rotation phase
              'theta': 180,  # 2d projection angle
              'phi': 0,  # Inclinaison line of sight
              'x0': 0,
              'y0': 0,
              'd_arm': step * 1e3 * dpc,
              'pixel_size': 1}

    param2 = param1.copy()
    param3 = param1.copy()
    param4 = param1.copy()

    param2['eta'] = eta1 + alpha.nominal_value - alpha.std_dev
    param3['eta'] = eta1 + alpha.nominal_value
    param4['eta'] = eta1 + alpha.nominal_value + alpha.std_dev

    t = np.linspace(0, turn * 2 * np.pi, npts)
    u_1, v_1 = spiral_archimedian(t, param1)
    u_2, v_2 = spiral_archimedian(t, param2)
    u_3, v_3 = spiral_archimedian(t, param3)
    u_4, v_4 = spiral_archimedian(t, param4)

    fl1 = interp1d(v_1, u_1, bounds_error=False, kind='linear',
                   fill_value=0)
    fl2 = interp1d(v_2, u_2, bounds_error=False, kind='linear',
                   fill_value=0)
    fl3 = interp1d(v_3, u_3, bounds_error=False, kind='linear',
                   fill_value=0)
    fl4 = interp1d(v_4, u_4, bounds_error=False, kind='linear',
                   fill_value=0)

    e_cut_inf = (fl1(pos_cut) - fl2(pos_cut))
    e_cut_sup = (fl1(pos_cut) - fl4(pos_cut))

    arm_cut = abs(np.mean([e_cut_inf, e_cut_sup]))
    e_arm_cut = abs(np.std([e_cut_inf, e_cut_sup]))

    s_arm = ufloat(arm_cut, e_arm_cut)

    if verbose:
        print('s_arm = %2.1f +/- %2.1f AU' % (arm_cut, e_arm_cut))
    if display:
        plt.figure(figsize=[4, 4])
        plt.plot(u_1, v_1, color='tab:blue',
                 label=r'$\alpha$ = %2.1f deg' % alpha.nominal_value)
        plt.plot(u_2, v_2, color='tab:orange', ls='--')
        plt.plot(u_3, v_3, color='tab:blue')
        plt.plot(u_4, v_4, color='tab:orange', ls='--')
        plt.plot([fl1(pos_cut), fl3(pos_cut)], [pos_cut, pos_cut], '.-',
                 color='crimson',
                 label='s = %2.1f+/-%2.1f AU' % (arm_cut, e_arm_cut))
        plt.xlabel('X [AU]')
        plt.ylabel('Y [AU]')
        plt.legend(fontsize=9)
        plt.subplots_adjust(top=0.97, left=0.17, bottom=0.12, right=0.94)
    return s_arm


def ComputeChi2r(model, X, Y, e, param):
    m = model(X, *param)
    chi21 = ((Y - m)**2) / e**2
    chi2_t = chi21.sum()
    chi2_red = chi2_t / (float(len(m) - len(param)))
    return chi2_red


def fit_param_hydro(x_model, y_model, m_type, param_name, model_axis,
                    rel_err=10, split=False, lim=None, absolute_sigma=True,
                    save=True, scale_err=1, verbose=True):
    """ Fit the parameter by a m_type model.

    Parameters
    ----------
    `x_model` : {array}
        Input of model,\n
    `y_model` : {array}
        Data to fit,\n
    `m_type` : {function}
        Model to fit,\n
    `param_name` : {str}
        Name of the fitted parameters,\n
    `rel_err` {float}:
        If no error in input model, uncertainties are set as relative uncertainties
        with y_model * rel_err [%],\n
    `model_axis` : {array}
        Input to plot the fitted model,\n
    `split` : {bool}, (optional)
        If True, data are splitted into 2 parts by lim, by default False,\n
    `lim` : {float}, (optional)
        Limit use to split the data set (of input axis), by default None,\n
    `absolute_sigma` : {bool}, (optional)
        If True, use uncertainties on data as absolute (not normalize by chi2), by default True,\n
    `verbose` : {bool}, (optional)
        If True, print the informations, by default True.

    Returns
    -------
    `dic1`, `dic2`: {dict}
        Dictionnary with fitted parameters (`a`, `b`), uncertainties (`e_a`, `e_b`), limite of the
        regim (`lim_regim`), and fitted model (`mod`, mod[0]: model, mod[1]: lower limit of the
        model, mod[2]: upper limit of the model).
    """
    if split:
        if lim == 0:
            regim_1 = x_model > lim
            regim_2 = x_model > lim
        else:
            regim_1 = x_model <= lim
            regim_2 = x_model > lim

        # Part 1 if two regim
        X = x_model[regim_1]
        Y = y_model[:, 0][regim_1]
        e_Y = y_model[:, 1][regim_1] * scale_err
        popt1, pcov1 = curve_fit(
            m_type, X, Y, sigma=e_Y, absolute_sigma=absolute_sigma)

        # Part 2 if two regim
        X2 = x_model[regim_2]
        Y2 = y_model[:, 0][regim_2]
        e_Y2 = y_model[:, 1][regim_2] * scale_err
        popt2, pcov2 = curve_fit(
            m_type, X2, Y2, sigma=e_Y2, absolute_sigma=absolute_sigma)

        p_sigma1 = np.sqrt(np.diag(pcov1))
        p_sigma2 = np.sqrt(np.diag(pcov2))

        chi2_red_1 = ComputeChi2r(m_type, X, Y=Y, e=e_Y, param=popt1)
        chi2_red_2 = ComputeChi2r(m_type, X2, Y=Y2, e=e_Y2, param=popt2)

        if verbose:
            print('\n%s, regime 1: a = %2.2f +/- %2.2f, b = %2.1f +/- %2.1f, chi2_r = %2.2f' %
                  (param_name, popt1[0], p_sigma1[0], popt1[1], p_sigma1[1], chi2_red_1))
            print('%s, regime 2: a = %2.2f +/- %2.2f, b = %2.1f +/- %2.1f, chi2_r = %2.2f' %
                  (param_name, popt2[0], p_sigma2[0], popt2[1], p_sigma2[1], chi2_red_2))

        a = model_linear(model_axis, *popt1)
        mod_inf = model_linear(model_axis, *popt1 - p_sigma1)
        mod_sup = model_linear(model_axis, *popt1 + p_sigma1)
        mod1 = [a, mod_inf, mod_sup]

        a = model_linear(model_axis, *popt2)
        mod_inf = model_linear(model_axis, *popt2 - p_sigma2)
        mod_sup = model_linear(model_axis, *popt2 + p_sigma2)
        mod2 = [a, mod_inf, mod_sup]

        dic1 = {'a': popt1[0], 'b': popt1[1], 'e_a': p_sigma1[0],
                'e_b': p_sigma1[1], 'lim_regim': lim, 'mod': mod1,
                'x_mod': model_axis}
        dic2 = {'a': popt2[0], 'b': popt2[1], 'e_a': p_sigma2[0],
                'e_b': p_sigma2[1], 'lim_regim': lim, 'mod': mod2,
                'x_mod': model_axis}
        file = open('SavedDpy/fit_%s.dpy' % param_name, 'w+')
        pickle.dump([dic1, dic2, model_axis], file, 2)
        file.close()
        return dic1, dic2
    else:
        X = x_model
        try:
            Y = y_model[:, 0]
            e_Y = y_model[:, 1]
        except Exception:
            Y = y_model
            e_Y = y_model * rel_err / 100.

        # if param_name == 'mass':
        #     absolute_sigma = False
        popt1, pcov1 = curve_fit(m_type, X, Y, sigma=e_Y,
                                 absolute_sigma=absolute_sigma)
        p_sigma1 = np.sqrt(np.diag(pcov1))
        try:
            chi2_red_1 = ComputeChi2r(m_type, X, Y=Y, e=e_Y, param=popt1)
        except Exception:
            chi2_red_1 = np.nan

        if verbose:
            print('\n%s, regime 1: a = %2.2e+/- %2.2e, b = %2.1e +/- %2.1e, chi2_r = %2.2f' %
                  (param_name, popt1[0], p_sigma1[0], popt1[1], p_sigma1[1], chi2_red_1))

        a = model_linear(model_axis, *popt1)
        mod_inf = model_linear(model_axis, *popt1 - p_sigma1)
        mod_sup = model_linear(model_axis, *popt1 + p_sigma1)
        mod1 = [a, mod_inf, mod_sup]

        dic1 = {'a': popt1[0], 'b': popt1[1], 'e_a': p_sigma1[0],
                'e_b': p_sigma1[1], 'lim_regim': lim, 'mod': mod1}
        if save:
            file = open('SavedDpy/fit_%s.dpy' % param_name, 'w+')
            pickle.dump([dic1, model_axis], file, 2)
            file.close()
        return dic1


def _compute_chi2r(model, X, Y, e, param):
    m = model(X, *param)
    chi2 = ((Y - m)**2) / e**2
    chi2_t = chi2.sum()
    chi2_red = chi2_t / (float(len(m) - len(param)))
    return chi2_red


def fit_linear_Mdust_rnuc(l_Mdust, l_mix, l_rnuc, f_norm=3e7,
                          scale_a=1, scale_b=1, ioutput=270,
                          absolute_sigma=True,
                          verbose=True, display=True):
    N = ioutput * 0.00125
    mix_log = np.log(np.logspace(-3, 2, 300))
    la, l_ea, lb, l_eb = [], [], [], []
    for i in range(len(l_Mdust)):
        M_dust2 = fit_param_hydro(np.log(l_mix), l_Mdust[i], model_linear,
                                  'mass', mix_log, split=False, lim=5,
                                  save=False, verbose=False)
        la.append(M_dust2['a'] / N)
        l_ea.append(M_dust2['e_a'] * scale_a)
        lb.append(M_dust2['b'] / N)
        l_eb.append(M_dust2['e_b'] * scale_b)

    la, lb = np.array(la), np.array(lb)
    l_ea, l_eb = np.array(l_ea), np.array(l_eb)

    fit_ar, cov_ar = curve_fit(model_linear, l_rnuc, la, sigma=l_ea,
                               absolute_sigma=absolute_sigma)
    fit_br, cov_br = curve_fit(model_linear, l_rnuc, lb, sigma=l_eb,
                               absolute_sigma=absolute_sigma)
    e_ar = np.sqrt(np.diag(cov_ar))
    e_br = np.sqrt(np.diag(cov_br))

    chi2r_ar = _compute_chi2r(model_linear, l_rnuc,
                              Y=la, e=l_ea, param=fit_ar)
    chi2r_br = _compute_chi2r(model_linear, l_rnuc,
                              Y=lb, e=l_eb, param=fit_br)

    a, b = fit_ar[0] / f_norm, fit_ar[1] / f_norm
    c, d = fit_br[0] / f_norm, fit_br[1] / f_norm
    e_a, e_b = e_ar[0] / f_norm, e_ar[1] / f_norm
    e_c, e_d = e_br[0] / f_norm, e_br[1] / f_norm

    mdust_fit = {'a': a, 'b': b, 'c': c, 'd': d,
                 'e_a': e_a, 'e_b': e_b, 'e_c': e_c,
                 'e_d': e_d}

    if verbose:
        print('\na_r: a = (%2.3f +/- %2.3f) x %s, b = (%2.2f +/- %2.2f) x %s, chi2_r = %2.2f' %
              (a, e_a, str(f_norm), b, e_b, str(f_norm), chi2r_ar))
        print('b_r: a = (%2.3f +/- %2.3f) x %s, b = (%2.2f +/- %2.2f) x %s, chi2_r = %2.2f' %
              (c, e_c, str(f_norm), d, e_d, str(f_norm), chi2r_br))
        print('M_ext = N*xi*(ar*np.log(mix) + br))')

    if display:
        rnuc_mod = np.linspace(10, 40, 100)

        plt.figure()
        plt.errorbar(l_rnuc, la * f_norm, yerr=l_ea * f_norm * scale_a, marker='.',
                     ecolor='lightgrey', ls='')
        plt.errorbar(l_rnuc, lb * f_norm, yerr=l_eb * f_norm * scale_b, marker='.',
                     ecolor='lightgrey', color='orange', ls='')
        plt.plot(rnuc_mod, model_linear(rnuc_mod, * fit_ar) * f_norm, '--',
                 color=flatui[1], label=r'fit $a_r$',
                 alpha=.5)
        plt.plot(rnuc_mod, model_linear(rnuc_mod, * fit_br) * f_norm, '--',
                 color=flatui[3], label=r'fit $b_r$', alpha=.5)
        plt.xlim(rnuc_mod[0], rnuc_mod[-1])
        plt.xlabel('$r_{nuc}$ [AU]', fontsize=14)
        plt.ylabel('Linear coefficients', fontsize=14)
        plt.grid(alpha=.2, color='gray')
        plt.legend()

    return mdust_fit
