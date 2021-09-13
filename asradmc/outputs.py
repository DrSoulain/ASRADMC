# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:31:43 2019

@author: Anthony Soulain (University of Sydney)

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

Functions to read and save the outputs of radmc3d.

------------------------------------------------------------------------
"""

import os
import pickle
import time

import numpy as np
from astropy import constants
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from munch import munchify as dict2class
from termcolor import cprint

from asradmc.radmc3dPy import analyze
from asradmc.radmc3dPy import image as imageradmc
from asradmc.tools import rad2mas, write_fits

# Some natural constants
au = constants.au.cgs.value  # Astronomical Unit [cm]
pc = constants.pc.cgs.value  # Parsec            [cm]
ms = constants.M_sun.cgs.value


def _create_arb_save(namedir):
    """ Create the directories tree to save files. """
    if not os.path.exists("%s/" % namedir):
        os.system("mkdir %s" % namedir)
    return "%s/" % namedir


def _plot_histo_temperature(l_temp, Tsub=2000, Tmax=3500, Tstep=100):
    """ Plot histogram of the unstructured temperature list computed with
    radmc3d. 

    Parameters:
    -----------
    `l_temp` {list}:
        List of unstructured temperature (location unknown),\n
    `Tsub` {float}:
        Sublimation temperature [K],\n
    `Tmax` {float}:
        Maximum temperature to plot,\n
    `Tstep` {float}:
        Step of temperature for the histogram.
    """
    cond = l_temp >= 0
    n_dust = float(len(l_temp[cond]))

    Tmean = l_temp.mean()
    n_over_Tsub = len(l_temp[l_temp > Tsub])

    fig = plt.figure()
    plt.hist(l_temp[cond & (l_temp < Tsub)], bins=np.arange(0, Tmax, Tstep))
    plt.hist(l_temp[l_temp >= Tsub], bins=np.arange(0, Tmax, Tstep), color='crimson', alpha=.5,
             label='Hot cells = %i (%2.3f%%)' % (n_over_Tsub, 100 * (n_over_Tsub / n_dust)))
    plt.axvline(Tmean, label=r'T$_{mean}$ (%2.0f K)' % Tmean)
    plt.axvline(Tsub, linestyle='--',
                color='r', label=r'T$_{sub}$ (%2.0f K)' % Tsub)
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("T [K]", fontsize=12, color='gray')
    plt.ylabel("# cells", fontsize=12, color='gray')
    plt.yscale('log')
    plt.ylim(1, 1e7)
    plt.xlim(0, 3500)
    plt.tight_layout()
    return fig


def _plot_sed_radmc(sed, dpc):
    """ Plot sed (Jy vs. µm) using radmc3dpy analyze functions. """
    fluxnu = sed[:, 1]

    distfact = 1.0 / (dpc ** 2)
    f_jy = 1e23 * fluxnu * distfact

    vmax = 10 * f_jy.max()
    vmin = vmax / 1e4

    fig = plt.figure()
    analyze.plotSpectrum(sed, jy=True, ylg=True, xlg=True,
                         micron=True, dpc=dpc, obs=False,
                         fnu=False)
    plt.grid(which="both", alpha=0.2)
    plt.ylim([vmin, vmax])
    plt.xlim([0.05, 100])
    plt.xlabel(r"Wavelength [$\mu m$]", fontsize=12)
    plt.ylabel(r"F$_\nu$ [Jy]", fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def check_image_exist(npts, param, wl, loadlambda, wl_grid_ima,
                      perform_amr=False, savedir='',
                      model_type='ramses'):
    """Check if the image model already exist in savedir.

    Parameters
    ----------
    `npts` {int}:
        Size of the simulation in pixels,\n
    `param` {dict}:
        Parameters of the simulation,\n
    `wl` {float}:
        Wavelength of the computed image,\n
    `loadlambda` {bool}:
        If `loadlambda`==True, the image is a chromatic cube image,\n
    `wl_grid_ima` {list or str}:
        If `loadlambda`==True, wl_grid_ima can be the list
        of wavelength [µm] or a string corresponding to
        different wavelength (see `inputs._saved_wl_tab()`),\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `model_type` {str}:
        Type of model ('spiral' or 'ramses'),\n
    `savedir` {str}:
        Directory adress to save the results.

    """

    # Create the output directory
    if not len(savedir) == 0:
        if not os.path.exists(savedir):
            os.mkdir(savedir)

    # dpc need to be in [parsec], rnuc in mas
    rnuc = param["rnuc"] * param['dpc']  # in [AU]
    xi = param["xi"] * 100.0  # in [%]
    mix = param["mix"] * 100.0  # in [%]

    dirr = "%sSaved_images/rnuc=%2.2f/xi=%2.2f/" % (
        savedir, rnuc, xi)

    filename = _create_fitsname_radmc(model_type, param, npts,
                                      wl=wl, loadlambda=loadlambda,
                                      wl_grid_ima=wl_grid_ima,
                                      perform_amr=perform_amr)
    s_lambda = "wl=%2.1f" % wl
    if loadlambda:
        s_lambda = "cube"

    if os.path.exists(dirr + filename + '.fits'):
        cprint("\n---------------------------------------------------------", "cyan")
        cprint("Model %s (N=%i, xi=%2.2f, rnuc=%2.2f, mix=%2.2f) already done."
               % (s_lambda, npts, xi, rnuc, mix),
               "cyan")
        cprint("---------------------------------------------------------", "cyan")
        return True
    else:
        return False


def check_temp_exist(npts, param, perform_amr=False, savedir='',
                     verbose=False):
    """
    Check if the temperature computation is already done for the
    given parameters (rnuc, xi and mix).

    Parameters:
    -----------
    `npts` {int}:
        Size of the simulation in pixels,\n
    `param` {dict}:
        Parameters of the simulation,\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `savedir` {str}:
        Directory adress to save the results.
    """

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    tempdir = savedir + 'Saved_temp/'
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)

    list_rnuc2 = os.listdir(tempdir)
    list_rnuc3 = [float(x.split('=')[1]) for x in list_rnuc2 if 'rnuc' in x]

    list_xi = []
    for x in os.listdir(tempdir):
        if 'rnuc' in x:
            tmp = os.listdir(tempdir + x)
            tmp2 = [float(x.split('=')[1]) for x in tmp if 'xi' in x]
            list_xi.extend(tmp2)
    list_xi = list(set(list_xi))

    if perform_amr:
        ss = "AMR"
    else:
        ss = "NoAmr"

    rnuc = param['rnuc']
    xi = param['xi'] * 100
    mix = param['mix'] * 100
    dpc = param['dpc']

    rnuc_au = rnuc * dpc
    done = False
    if np.round(rnuc_au, 2) in list_rnuc3:
        if np.round(xi, 2) in list_xi:
            tmp = tempdir + 'rnuc=%2.2f/xi=%2.2f/' % (rnuc_au, xi)
            filename = tmp + 'Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s.fits' % (npts, xi, rnuc_au,
                                                                                      mix, ss)
            if os.path.exists(filename):
                cprint(
                    "\n---------------------------------------------------------", "cyan")
                cprint(
                    "Temp computation %s (N=%i, xi=%2.2f, rnuc=%2.2f, mix=%2.2f) already done."
                    % (ss, npts, xi, rnuc_au, mix),
                    "cyan",
                )
                cprint(
                    "---------------------------------------------------------", "cyan")
                done = True
            else:
                done = False

    if not done and verbose:
        cprint('\nAlready done models:', 'cyan')
        cprint('--------------------', 'cyan')
        print('rnuc: ', list_rnuc3)
        print('xi: ', list_xi)
        print('')
    return done


def _create_fitsname_radmc(model_type, param, npts, wl=None, perform_amr=False,
                           loadlambda=False, wl_grid_ima=None,
                           wl_grid_spec=None):
    """ Create the output fits name to save the computed image with radmc3d. 

    Parameters:
    -----------
    `model_type` {str}:
        Type of model ('spiral' or 'ramses'),\n
    `param` {dict}:
        Parameters of the simulation,\n
    `npts` {int}:
        Size of the simulation in pixels,\n
    `wl` {float}:
        Wavelength of the computed image,\n
    `loadlambda` {bool}:
        If `loadlambda`==True, the image is a chromatic cube image,\n
    `wl_grid_ima` {list or str}:
        If `loadlambda`==True, wl_grid_ima can be the list
        of wavelength [µm] or a string corresponding to
        different wavelength (see `inputs._saved_wl_tab()`),\n
    `wl_grid_spec` {str}:
        name of the list of wavelengths to compute the SED (see `inputs._saved_wl_tab()`),\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed.

    Outputs:
    --------
    fitsname {str}:
        Name of the saved file (contains important parameters (e.g.: rnuc, xi, 
        mix if `model_type` is 'ramses').
    """

    if wl_grid_spec is not None:
        str_wl = wl_grid_spec
    else:
        str_wl = "wl=" + str(wl)
        if loadlambda:
            str_wl = wl_grid_ima
            if type(wl_grid_ima) == list:
                str_wl = "cube"

    str_amr = "NoAmr"
    if perform_amr:
        str_amr = "AMR"

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]
    mix = param["mix"] * 100.0  # in [%]

    i = param["incl"]

    if model_type == "spiral":
        alpha = param["alpha"]
        omega = param["omega"] * 100
        fitsname = "%s_%s_npix=%d_" % (model_type, str_wl, npts)\
            + "xi=%2.2f_rnuc=%2.2f_mix=%2.2f_" % (xi, rnuc, mix)\
            + "alpha=%2.2f_o=%2.2f_i=%d_%s" % (float(alpha), omega, i, str_amr)
    elif model_type == "ramses":
        fitsname = "%s_%s_npix=%d_" % (model_type, str_wl, npts)\
            + "xi=%2.2f_rnuc=%2.2f_mix=%2.2f_" % (xi, rnuc, mix)\
            + "i=%d_%s" % (i, str_amr)
    else:
        fitsname = None
    return fitsname


def save_rho(npts, param, binary=True, perform_amr=False, savedir=''):
    """ Save the density grid used in radmc3d. 

    Parameters:
    -----------
    `npts` {int}:
        Size of the simulation in pixels,\n
    `param` {dict}:
        Parameters of the simulation,\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `binary` {bool}:
        If True, output and input are binary files,\n
     `savedir` {str}:
        Directory adress to save the results.

    Outputs: 
    --------
    `rho_save` {dict}:
        Saved dictionnary of dust density ('rho', 'Vol_cell' and 'mass_tot').
    """
    data = analyze.readData(ddens=True, dtemp=False, octree=perform_amr,
                            binary=binary)

    rho = data.rhodust
    try:
        rho2 = rho.reshape(len(rho))
    except ValueError:
        rho2 = rho.copy()

    rho_dust = rho2[rho2 > 1e-50]

    try:
        cellV = data.grid.getCellVolume()
    except ValueError:
        cellV_one = (2 * abs(data.grid.xi[0]) / data.grid.nx)**3
        cellV = np.ones(rho2.shape) * cellV_one

    cellV2 = cellV[rho2 > 1e-50]

    m = cellV2 * rho_dust

    rho_save = {'rho': rho_dust,
                'Vol_cell': cellV2,
                'mass_tot': m.sum() / ms
                }

    str_amr = "NoAmr"
    if perform_amr:
        str_amr = "AMR"

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]
    mix = param["mix"] * 100.0  # in [%]

    path = _create_arb_save("%sSaved_dens" % savedir)
    path = _create_arb_save(path + "rnuc=%2.2f" % rnuc)
    path = _create_arb_save(path + "xi=%2.2f" % xi)

    filename = "Rhodist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s" % (
        npts, xi, rnuc, mix, str_amr)

    file = open(path + filename + '.dpy', 'wb')
    pickle.dump(rho_save, file, 2)
    file.close()
    return rho_save


def _read_image_radmc(wl, dpc, Bmax=8.):
    """ Read radmc output image (in the local directory). 

    Parameters:
    -----------
    `wl` {float}:
        Wavelength of the computed image [µm],\n
    `dpc` {float}: 
        Astrometrical distance of the target [pc],\n
    `Bmax` {float}: 
        Maximum baseline to compute the convolved image [m].\n

    Outputs:
    --------
    `res_image` {class}:
        Results of radmc image computation (keys: 'cube', 'image', 
        'image_conv', 'flux', etc.). 
    """
    cwd = os.getcwd()
    im = imageradmc.readImage(cwd + '/image.out')

    im_wav = im.wav
    try:
        n35 = np.where(im_wav == wl)[0][0]
    except IndexError:
        n35 = 0
        wl = im.wav[n35]

    conv = ((im.sizepix_x) * (im.sizepix_y) / (dpc * pc)
            ** 2.0 * 1e23)  # Conversion factor to Jy

    image = im.image[:, :, n35] * conv  # Image [Jy]
    npts = image.shape[0]

    pixau = im.sizepix_x / au
    pix_size = pixau / (dpc / 1000.0)  # pixel size [mas]

    fov = pix_size * npts  # [mas]

    resol = rad2mas((wl * 1e-6) / (2 * Bmax)) / 1000.0
    cim = im.imConv(fwhm=[resol, resol], pa=0.0, dpc=dpc)

    if resol <= pix_size / 1000.0:
        image_conv = image.copy()
        print("#Warning : Resolution < pixel scale (no need of convolution).")
    else:
        image_conv = cim.image[:, :, n35] * conv  # Convolved image [Jy]

    cube = np.array([im.image[:, :, i] for i in range(len(im.wav))]) * conv

    flux = _compute_spectrum_image(cube)

    res_image = dict2class({'wav': im_wav, 'cube': cube,
                            'image': image, 'image_conv': image_conv,
                            'pix_size': pix_size, 'n35': n35, 'flux': flux,
                            'wl': wl, 'fov': fov, 'Bmax': Bmax})
    return res_image


def _compute_spectrum_image(cube):
    """ Integrate flux over the image to get the spectrum [Jy]. """
    n_wl = cube.shape[2]

    flux = []
    for i in range(n_wl):
        tmp = cube[:, :, i]
        tflux = tmp.sum()
        flux.append(tflux)
    flux = np.array(flux)
    return flux


def plot_image_results(res_image, dpc, unit='AU', log=False, p=0.5):
    """ Plot the resulting image (res_image from `_read_image_radmc()`)
    with dpc in [pc]. """
    image = res_image.image
    image_conv = res_image.image_conv

    image[image <= 0] = 0
    image_conv[image_conv <= 0] = 0

    im_rot = np.rot90(image)
    im_rot[im_rot <= 0] = 1e-20
    im_conv_rot = np.rot90(image_conv)
    im_conv_rot[im_conv_rot <= 0] = 1e-20

    # Extract the integrated flux
    tflux = image.sum()
    wl = res_image.wl

    fov_im = res_image.fov / 1000.

    # Convert to good spatial units
    if unit == "AU":
        xlabel, ylabel = "X", "Y"
        fact = dpc
    elif unit == "cm":
        fact = dpc * au
        xlabel = "X"
        ylabel = "Y"
    elif unit == "mas":
        xlabel = "RA offset"
        ylabel = "DEC offset"
        fact = 1000.
    else:
        unit = "arcsec"
        xlabel = "RA offset"
        ylabel = "DEC offset"
        fact = 1

    extent = np.array([fov_im / 2.0, -fov_im / 2.0, -
                       fov_im / 2.0, fov_im / 2.0]) * fact

    vmax = image.max()
    vmin = vmax / 1.e4

    norm = PowerNorm(p)
    if log:
        norm = LogNorm()

    fig = plt.figure(figsize=(13, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_title(r"Image ($\lambda$ = %2.2f $\mu m$), F$_{tot}$ = %2.4f Jy" % (wl, tflux),
                 fontsize=14,
                 color="Navy")
    im_cbar = ax.imshow(im_rot, norm=norm, cmap="afmhot", vmin=vmin, vmax=vmax,
                        origin="upper", extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    ax.set_xlabel("%s [%s]" % (xlabel, unit), fontsize=12, color="gray")
    ax.set_ylabel("%s [%s]" % (ylabel, unit), fontsize=12, color="gray")
    clb = plt.colorbar(im_cbar, cax=cax)
    # clb.set_label('Flux density [Jy]', fontsize = 10, color = 'k')
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_title(
        r"Image convolved ($\theta$ = $\lambda$/2B, B = %2.0f m)" % (res_image.Bmax),
        fontsize=14,
        color="Navy")
    im_cbar = ax.imshow(im_conv_rot, norm=norm, cmap="afmhot", vmin=vmin, vmax=vmax,
                        origin="upper", extent=extent)
    divider = make_axes_locatable(ax)
    ax.set_yticks([])
    ax.set_xlabel("%s [%s]" % (xlabel, unit), fontsize=12, color="gray")
    cax = divider.append_axes("right", size="3%", pad=0.05)
    clb = plt.colorbar(im_cbar, cax=cax)
    clb.set_label("Flux density [Jy]")
    plt.tight_layout()
    plt.show(block=False)
    return fig


def save_image_results(wl, dpc, param, wl_grid_ima, loadlambda,
                       model_type="spiral", savedir="",
                       perform_amr=False, save=False,
                       display=False):
    """ 
    Description: 
    ------------

    Save radmc image results. 

    Parameters:
    -----------

    `wl` {float}:
        Wavelength of the displayed image [µm],\n
    `dpc` {float}:
        Distance of the object in [pc],\n
    `param` {dict}:
        Parameters of the simulation,\n
    `wl_grid_ima` {list or str}:
        If `loadlambda`==True, wl_grid_ima can be the list
        of wavelength [µm] or a string corresponding to
        different wavelength (see `inputs._saved_wl_tab()`),\n
    `loadlambda` {bool}:
        If `loadlambda`==True, the image is a chromatic cube image,\n
    `model_type` {str}:
        Type of model ('spiral' or 'ramses'),\n
    `savedir` {str}:
        Directory adress to save the results,\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `save` {bool}:
        If True, results are saved,\n

    Outputs:
    --------
    `res_image` {class}:
        Results of radmc image computation (keys: 'cube', 'image', 
        'image_conv', 'flux', etc.). 
    """

    res_image = _read_image_radmc(wl, dpc, Bmax=130)

    npts = res_image.image.shape[0]

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]

    model_dir = ''
    if save:
        model_dir = _create_arb_save("%sSaved_images/" % savedir)
        model_dir = _create_arb_save(model_dir + "rnuc=%2.2f" % rnuc)
        model_dir = _create_arb_save(model_dir + "xi=%2.2f" % xi)

    fitsname = _create_fitsname_radmc(model_type, param, npts,
                                      wl=wl, loadlambda=loadlambda,
                                      wl_grid_ima=wl_grid_ima,
                                      perform_amr=perform_amr)

    param["wl_grid"] = wl_grid_ima

    if save:
        write_fits(model_dir + fitsname + ".fits",
                   res_image.cube, res_image.wav, res_image.pix_size,
                   param, align=-1)
    return res_image


def save_sed_results(npts, wl_grid_spec, param=None,
                     model_type="spiral", savedir='',
                     perform_amr=False, save=False,
                     display=False):
    """ Read and save SED results from radmc-3d. 

    Parameters:
    -----------
    `npts` {int}:
        Size of the simulation in pixels,\n
    `wl_grid_spec` {str}:
        name of the list of wavelengths to compute the SED
        (see `inputs._saved_wl_tab()`),\n
    `param` {dict}:
        Parameters of the simulation,\n
    `model_type` {str}:
        Type of model ('spiral' or 'ramses'),\n
    `savedir` {str}:
        Directory adress to save the results,\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `save` {bool}:
        If True, results are saved,\n

    Outputs:
    --------
    `sed_save` {dic}:
        Saved SED dictionnary (keys: 'flux' [Jy], 'wl' [µm] and
        'param').
    """

    sed = analyze.readSpectrum()

    dpc = param['dpc']

    lam = sed[:, 0]
    fluxnu = sed[:, 1]

    distfact = 1.0 / (dpc ** 2)

    f_jy = fluxnu * 1e23 * distfact

    if display:
        _plot_sed_radmc(sed, dpc)

    dic = {"flux": f_jy, "wl": lam, "param": param}

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]

    fitsname = _create_fitsname_radmc(model_type, param, npts,
                                      wl_grid_spec=wl_grid_spec,
                                      perform_amr=perform_amr)

    if save:
        path = _create_arb_save("%sSaved_spec" % savedir)
        path = _create_arb_save(path + "rnuc=%2.2f" % rnuc)
        path = _create_arb_save(path + "xi=%2.2f" % xi)
        file = open(path + fitsname, "wb")
        pickle.dump(dic, file)
        file.close()
    return dic


def save_temp_results(npts, param, start_time=0, Tsub=2000,
                      binary=True, rto_style=3, savedir="",
                      perform_amr=False, save=False,
                      display=False):
    """ Read and save unstructured (cell location unknown) temperature
    distribution.

    Parameters:
    -----------
    `npts` {int}:
        Size of the simulation in pixels,\n
    `param` {dict}:
        Parameters of the simulation,\n
    `start_time` {float}:
        Initial time to save computing time,\n
    `Tsub` {float}:
        Sublimation temperature [K],\n
    `binary` {bool}:
        If True, output and input are binary files,\n
    `rto_style` {int}:
        If 3, outputs are binary files,\n
    `savedir` {str}:
        Directory adress to save the results,\n
    `perform_amr` {bool}:
        If True, amr grid refinement is performed,\n
    `save` {bool}:
        If True, results are saved.

    Outputs:
    --------
    `save_temp` {dict}:
        Temperature results obtained with radmc3d (keys: "Tmax", "Tmean",
        "Nhot", "param" and "time").
    """

    # Read the density grid (to localise the dust)
    if binary:
        dens = np.fromfile("dust_density.binp")
    else:
        dens = np.fromfile("dust_density.inp")

    # Read the dust temperature results
    if rto_style == 3:
        temp = np.fromfile("dust_temperature.bdat")
    else:
        temp = np.fromfile("dust_temperature.dat")

    # Find the dust in the grid
    loc_dust = dens > 1e-40

    l_temp = temp[loc_dust]

    Tmean = l_temp.mean()
    Tmax = l_temp.max()

    # Compute the number of cells above the sublimation temperature
    n_over_Tsub = len(l_temp[l_temp > Tsub])

    print("\nTmean = %2.0f K; Tmax = %2.0f K, Hot cells %i\n" %
          (Tmean, Tmax, n_over_Tsub))

    if display:
        _plot_histo_temperature(l_temp, Tsub=Tsub)

    t = time.time() - start_time
    m = t / 60.0

    # Dictionnary to be saved
    cond = l_temp >= 0
    n_dust = float(len(l_temp[cond]))
    n_over_rel = 100 * (n_over_Tsub / n_dust)
    save_temp = {"Tmax": Tmax, "Tmean": Tmean, "Nhot_rel": n_over_rel,
                 "Nhot": n_over_Tsub, "param": param, "time": m}

    s = "NoAmr"
    if perform_amr:
        s = "AMR"

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]
    mix = param["mix"] * 100.0  # in [%]

    path = _create_arb_save("%sSaved_temp" % savedir)
    path = _create_arb_save(path + "rnuc=%2.2f" % rnuc)
    path = _create_arb_save(path + "xi=%2.2f" % xi)

    filename = "Tdist_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s" % (
        npts, xi, rnuc, mix, s)

    if save:
        fits.writeto(path + filename + ".fits", l_temp, overwrite=True)
        file = open(path + filename + ".dpy", "wb")
        pickle.dump(save_temp, file)
        file.close()

    return save_temp


def plot_temp_map(data, nint, fov, dpc, savefits=False):
    """ Interpolate the amr grid temperature and dust
    density onto a linear one. Plot the temperature
    map in the middle plan with an overlap of the dust density
    contours.

    Parameters:
    -----------
    `data` {class}: 
        Class like object from analyze.readData of radmc3dpy (e.g: d =
        analyze.readData(ddens=True, dtemp=True, octree=True, binary=True),\n
    `nint` {int}:
        Size of the interpolated grid (image.shape=(nint, nint),\n
    `fov` {float}:
        Field of view of the grid [arcsec],\n
    `dpc` {float}:
        Astrometrical distance of the target [pc], use to compute the
        total size of the grid.\n

    Returns:
    --------
    `image_temp` {array}: 
        Interpolated temperature map,\n
    `image_rho` {array}:
        Interpolated dust density map,
    """
    xlim = np.array([-fov / 2., fov / 2.]) * dpc
    ylim = np.array([-fov / 2., fov / 2.]) * dpc

    nx = ny = nint
    nz = 1

    plot_x = xlim[0] + (xlim[1] - xlim[0]) * \
        np.arange(nx, dtype=float) / float(nx - 1)
    plot_y = ylim[0] + (ylim[1] - ylim[0]) * \
        np.arange(ny, dtype=float) / float(ny - 1)
    plot_z = ylim[0] + (ylim[1] - ylim[0]) * \
        np.arange(ny, dtype=float) / float(ny - 1)

    x, y, z = plot_x * au, plot_y * au, plot_z * au
    npoints = nx * ny * nz

    idata = {}
    ndust = data.rhodust.shape[1]
    idata["rhodust"] = np.zeros([nx, ny, nz, ndust], dtype=np.float64)
    idata["dusttemp"] = np.zeros([nx, ny, nz, ndust], dtype=np.float64)
    idata["cellID"] = np.zeros(npoints, dtype=np.int)

    cellID = None
    for ind in range(npoints):
        ix = int(np.floor(ind / ny / nz))
        iy = int(np.floor((ind - ix * ny * nz) / nz))
        iz = nx // 2
        try:
            cellID = data.grid.getContainerLeafID([x[ix], y[iy], z[iz]])
        except AttributeError:
            cellID = None

        if cellID is not None:
            idata["cellID"][ind] = cellID
            if cellID >= 0:
                ind_leaf = data.grid.leafID[cellID]
                idata["rhodust"][ix, iy, :] = data.rhodust[ind_leaf]
                idata["dusttemp"][ix, iy, :] = data.dusttemp[ind_leaf]

    if cellID is None:
        map_temp = data.dusttemp[:, :, :, 0]
        map_rho = data.rhodust[:, :, :, 0]
        image_temp = np.rot90(map_temp[:, :, ny // 2])
        image_rho = np.rot90(map_rho[:, :, ny // 2])
    else:
        map_temp = np.squeeze(idata["dusttemp"][:, :, :, 0])
        map_rho = np.squeeze(idata["rhodust"][:, :, :, 0])
        image_temp = np.rot90(map_temp[:, :, ])
        image_rho = np.rot90(map_rho[:, :, ])

    loc_dust = image_rho >= 1e-30

    #  [1:-1, 1:-1]
    plt.figure()
    plt.contourf(image_temp, levels=[0, 1000, 1500, 2000, 2500,
                                     3000, image_temp.max()],
                 origin='upper',
                 colors=("Navy", "dodgerblue", "skyblue",
                         "gold", "crimson", "purple"))
    cb = plt.colorbar()
    cb.set_label("T [K]")
    plt.contour(loc_dust, 1, colors=["k"],
                linewidths=[1], origin='upper')
    return image_temp, image_rho


def save_vtk(npts, param, perform_amr=False,
             setthreads=12, savedir=''):
    """ Save the 3d vtk file (3d grid with amr information, if any)."""
    if not os.path.exists(savedir):
        os.system('mkdir %s' % savedir)

    s_amr = 'NoAmr'
    if perform_amr:
        s_amr = 'AMR'

    os.system('radmc3d vtk_dust_density 1 setthreads %i' % setthreads)

    rnuc = param["rnuc"] * 1000.0 * param['dpc'] / 1000.  # in [AU]
    xi = param["xi"] * 100.0  # in [%]
    mix = param["mix"] * 100.0  # in [%]

    path = _create_arb_save("%sSaved_vtk" % savedir)
    path = _create_arb_save(path + "rnuc=%2.2f" % rnuc)
    path = _create_arb_save(path + "xi=%2.2f" % xi)

    filevtk = "vtk_npix=%i_xi=%2.2f_rnuc=%2.2f_mix=%2.2f_%s" % (
        npts, xi, rnuc, mix, s_amr)

    os.system('cp model.vtk %s.vtk' % filevtk)
    if os.path.exists('model.vtk'):
        os.remove('model.vtk')
    return None
