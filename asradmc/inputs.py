# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:31:43 2019

@author: Anthony Soulain (University of Sydney)

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

Function to generate inputs for radmc3d.

------------------------------------------------------------------------
"""

import os
import pickle
import struct

import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
from tqdm import tqdm

from asradmc.stellar_models import compute_luminosity, select_star_model


def _saved_wl_tab(inp):
    """ Load the saved list of wavelength for different 
    instruments (matisse_LM, matisse_N, gravity, pionier),
    different type of spectra (fastspectum, fastspectum#X)
    or image cube (VHKLN).
    """
    list_choice = ["matisse_LM", "matisse_N", "gravity", 
                   "pionier"]

    if type(inp) == list:
        wl_tab = inp
    else:
        if inp == "VHKLN":
            wl_tab = [0.55, 1.57, 2.27, 3.50, 10.0]
        elif inp == "spectrum":
            wl_tab = [0.4, 0.52, 0.62, 0.78, 0.9, 0.97, 0.99, 1.0, 1.03,
                      1.1, 1.21, 1.57, 2.27, 3.5, 4.5, 5.8, 6.5, 8.4, 12, 15, 20, 40]
        elif inp == "fastspectum":
            wl_tab = [0.4, 0.8, 1.21, 1.57, 2.27,
                      3.5, 4.5, 6.5, 10, 40]
        elif inp == "fastspectum2":
            wl_tab = [0.96873925, 1.19086373, 1.46391965, 1.7995852,
                      2.21221629, 2.71946052, 3.34301197, 4.10953897, 5.05182475,
                      5.6, 6.21016942, 7, 7.63411364, 9.38455736, 11.5363644, 14.18156428,
                      17.43328822, 21.43060752, 26.34448149, 32.38506907, 39.81071706]
        elif inp == "fastspectum3":
            wl_tab = [0.8, 2.4, 3.5, 4.5, 6.5, 10, 20, 40]
        elif inp == "fastspectum4":
            wl_tab = [2.3, 3, 5, 6.5, 8, 10, 13, 17, 20, 27]
        elif inp == "fastspectum5":
            wl_tab = [2.4, 3.5, 4.7, 6.5, 10, 15, 27]
        elif inp == "matisse_LM":
            wl_tab = np.linspace(2.83876, 4.13905, 13)
        elif inp == "matisse_N":
            wl_tab = np.linspace(7.45949, 13.69251, 20)
        elif inp == "gravity":
            wl_tab = np.linspace(1.97272, 2.24727, 11)
        elif inp == "pionier":
            wl_tab = np.linspace(1.533, 1.772, 6)

        else:
            print("If wl_grid_ima is not a list of wavelength, it must be :")
            print("-> image = V, J, H, K, L and N band,")
            print("-> spectrum = optimal wavelength for SED computation.")
            print("-> fastspectrum = small wavelength grid for SED computation.")
            print("-> instruments:", list_choice)
            wl_tab = []

    return wl_tab


def _plot_input_stellar_source(sed_star, types):
    lam = sed_star['lam']
    f_star1 = sed_star['fstar1']
    f_star2 = sed_star.get('fstar2', None)

    fig = plt.figure(figsize=(7, 5))
    plt.loglog(lam, f_star1, linewidth=1, label="star1 (%s)" %
               types[0], linestyle="-")
    if f_star2 is not None:
        plt.loglog(lam, f_star2, linewidth=1, label="star2 (%s)" % types[1])
        plt.loglog(lam, f_star1 + f_star2, "k--",
                   linewidth=1, label="Total SED")
    plt.xlim([np.min(lam), 1e2])
    plt.ylim([np.max(f_star1) * 1e-7, 1e1 * np.max(f_star1)])
    plt.ylabel(
        r"Spectral irradiance [$\rm{erg.cm^{-2}.s^{-1}.Hz^{-1}}$]", fontsize=11)
    plt.xlabel(r"Wavelength [$\mu m$]", fontsize=11)
    plt.grid(which="both", alpha=0.2)
    plt.legend()
    plt.subplots_adjust(top=0.995, bottom=0.11,
                        left=0.105, right=0.98,
                        hspace=0.2, wspace=0.2)
    return fig


def clean_inputs():
    """ Clean all previous inputs for radmc run. """
    list_input = ["dust_density.inp", "dust_density.binp", "amr_grid.inp",
                  "dustopac.inp", "radmc3d.inp", "amr_grid.inp", "stars.inp",
                  "wavelength_micron.inp", "camera_wavelength_micron.inp",
                  "dust_density_grid.txt", "amr_grid_noOct.txt", 'dust_density_grid.dpy',
                  ]
    for filename in list_input:
        if os.path.exists(filename):
            os.remove(filename)
    return None


def clean_outputs():
    """ Clean all previous outputs of radmc run. """
    list_input = ["dust_temperature.bdat",
                  "image.out", "spectrum.out", "radmc3d.out"]

    for filename in list_input:
        if os.path.exists(filename):
            os.remove(filename)
    return None


def make_camera_inp(inp):
    """ Create the camera.inp for radmc-3d. """
    wl_tab = _saved_wl_tab(inp)
    with open("camera_wavelength_micron.inp", "w+") as f:
        f.write("%i\n" % (len(wl_tab)))
        for wl in wl_tab:
            f.write("%2.3f\n" % (wl))
    return len(wl_tab)


def make_dustopac(name_dust_type, verbose=True):
    """ Make dustopac.inp for radmc-3d."""
    with open("dustopac.inp", "w+") as f:
        f.write("2               Format number of tshis file\n")
        f.write("1               Nr of dust species\n")
        f.write(
            "============================================================================\n")
        f.write("1               Way in which this dust species is read\n")
        f.write("0               0=Thermal grain\n")
        f.write(
            "%s        Extension of name of dustkappa_***.inp file\n" % name_dust_type

        )
        f.write(
            "----------------------------------------------------------------------------\n"
        )
        if verbose:
            cprint("##### dustopac.inp file created...\n", color="cyan")
    return None


def make_wavelength_grid(lam1=0.01, lam2=5.0, lam3=30.0, lam4=1e3, n12=50, n23=50, n34=15,
                         verbose=True):
    """
    Create list of wavelength used in RADMC3D
    (with default values: 0.01-5; 5-30; 30-1000 microns).

    Returns:
    ------------

    lam {array} : liste of lambda computed in different ranges [m].\n
    nlam {float}: Number of points in the wavelength grid.\n
    """
    lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
    lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
    lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
    lam = np.concatenate([lam12, lam23, lam34])
    nlam = lam.size

    # Write the wavelength file
    with open("wavelength_micron.inp", "w+") as f:
        f.write("%d\n" % (nlam))
        f.writelines("%13.6e\n" % l for l in lam)
        # for value in lam:
        #     f.write("%13.6e\n" % (value))

    if verbose:
        cprint("##### wavelength_micron.inp file created...", color="cyan")
    return lam, nlam


def make_radmc3d_param(nphot, nphot_scat, rto_style, verbose=True):
    """ Make radmc3d.inp for radmc-3d."""
    with open("radmc3d.inp", "w+") as f:
        f.write("nphot = %d\n" % (nphot))
        # Put this to 1 for isotropic scattering
        f.write("scattering_mode_max = 1\n")
        f.write(
            "nphot_scat = %d\n" % (nphot_scat)
        )  # Number of photon packet used for ray-tracing
        f.write("iranfreqmode = 1\n")
        f.write("rto_style = %d\n" % rto_style)
    if verbose:
        cprint("##### radmc3d.inp file created...\n", color="cyan")
    return None


def make_stars_input(dic, lam, nlam, types=["bb"], display=False,
                     verbose=True):
    """ Make stars.inp for radmc-3d."""
    nstar = len(list(dic.keys()))
    if not nstar == len(types):
        cprint("-------- WARNING -------", "r")
        print("nstar = 2 => need 2 types of SED")
        return None

    mstar, rstar, pstar = dic["star1"]["mstar"], dic["star1"]["rstar"], dic["star1"]["pstar"]
    mstar2 = rstar2 = pstar2 = None

    if nstar == 1:
        if verbose:
            print(("\n- STAR 1 %s" % types[0]))
        f_star1 = select_star_model(lam, dic["star1"], types[0],
                                    verbose=verbose)
        sed_star = {"fstar1": f_star1, "lam": lam}
        f_star2 = None
    elif (nstar == 2):
        if verbose:
            print(("\n- STAR 1 %s" % types[0]))
        f_star1 = select_star_model(lam, dic["star1"], types[0],
                                    verbose=verbose)
        if verbose:
            print(("\n- STAR 2 %s" % types[1]))
        f_star2 = select_star_model(lam, dic["star2"], types[1],
                                    verbose=verbose)
        sed_star = {"fstar1": f_star1, "fstar2": f_star2, "lam": lam}
        mstar2, rstar2 = dic["star2"]["mstar"], dic["star2"]["rstar"]
        pstar2 = dic["star2"]["pstar"]
    else:
        return None

    Ltot = compute_luminosity(sed_star, types)
    if display:
        _plot_input_stellar_source(sed_star, types)

    # Write the stars.inp file
    with open("stars.inp", "w+") as f:
        f.write("2\n")
        f.write("%s %d\n\n" % (nstar, nlam))
        if nstar == 1:
            f.write("%13.6e %13.6e %13.6e %13.6e %13.6e\n\n"
                    % (rstar, mstar, pstar[0], pstar[1], pstar[2]))
        elif nstar == 2:
            f.write("%13.6e %13.6e %13.6e %13.6e %13.6e\n"
                    % (rstar, mstar, pstar[0], pstar[1], pstar[2]))
            f.write("%13.6e %13.6e %13.6e %13.6e %13.6e\n\n"
                    % (rstar2, mstar2, pstar2[0], pstar2[1], pstar2[2]))
        f.writelines("%13.6e\n" % l for l in lam)
        f.write("\n")

        if (nstar == 1) & (f_star1 is not None):
            f.writelines("%13.6e\n" % l for l in f_star1)
        elif (nstar == 2) & (f_star1 is not None) & (f_star2 is not None):
            f.writelines("%13.6e\n" % l for l in f_star1)
            f.writelines("%13.6e\n" % l for l in f_star2)
        f.write("\n")

    if verbose:
        cprint("\n##### star.inp file created...\n", color="cyan")
    return sed_star, Ltot


def make_amr_grid(gridinfo, verbose=True):
    """ Create the amr_grid.inp for radmc-3d. """
    nx, ny, nz = gridinfo.nx, gridinfo.ny, gridinfo.nz
    xi, yi, zi = gridinfo.xi, gridinfo.yi, gridinfo.xi

    with open("amr_grid.inp", "w+") as f:
        f.write("1\n")  # iformat
        f.write("0\n")  # AMR grid style  (0=regular grid, no AMR)
        f.write("0\n")  # Coordinate system
        f.write("0\n")  # gridinfo
        f.write("1 1 1\n")  # Include x,y,z coordinate
        f.write("%d %d %d\n" % (nx, ny, nz))  # Size of grid
        f.writelines("%13.6e\n" % l for l in xi)
        f.writelines("%13.6e\n" % l for l in yi)
        f.writelines("%13.6e\n" % l for l in zi)
    
    if verbose:
        cprint("##### amr_grid.inp file created...\n", color="cyan")
    return None


def make_density_file(rhod_dust, binary=True, verbose=True):
    """ Create the density.inp for radmc-3d. """
    data = rhod_dust.ravel(order="F")
    
    if not binary:
        filename = "dust_density.inp"
        with open(filename, "w+") as f:
            f.write("1\n")  # Format number
            f.write("%d\n" % (len(data)))  # Nr of cells
            f.write("1\n")  # Nr of dust species
            # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep="\n", format="%13.6e")
            f.write("\n")
    else:
        filename = "dust_density.binp"
        # Create an array of integers for the header
        idum = np.array([1, 8, len(data), 1], dtype="int64")
        # First 1 is the format number, 8 means: double precision
        # ncells = nr of cells, last 1 means 1 dust species
        strf = str(len(data)) + "d"  # Create a format string for struct
        # Create a 1-D view, fortran-style indexing
        # Create a binary image of the data
        sdat = struct.pack("4Q", *idum) + struct.pack(strf, *data)
        with open(filename, "w+b") as f:
            f.write(sdat)

    if verbose:
        cprint("##### %s file created...\n" % (filename), color="cyan")
    return data


def make_amr_grid_octree(nx, ny, nz, xi, yi, zi, levelmax=1, nleaf=100,
                         nbranch=100, octree=np.zeros(1)):
    with open("amr_grid.inp", "w+") as f:
        f.write("1\n\n")  # iformat
        f.write("1\n")  # AMR grid style  (0=regular grid, no AMR)
        f.write("0\n")  # Coordinate system
        f.write("0\n\n")  # gridinfo
        f.write("1 1 1\n")  # Include x,y,z coordinate
        f.write("%d %d %d\n\n" % (nx, ny, nz))  # Size of grid
        f.write("%d %d %d\n\n" %
                (levelmax, nleaf, nbranch))  # Size of grid
        f.writelines("%13.6e\n" % l for l in xi)
        f.write("\n")
        f.writelines("%13.6e\n" % l for l in yi)
        f.write("\n")
        f.writelines("%13.6e\n" % l for l in zi)
        f.write("\n\n")
        f.writelines("%s\n" % l for l in tqdm(octree, desc='Saving amr_grid.inp',
                                              ncols=100, leave=False))
    return None
