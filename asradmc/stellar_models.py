#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:29:02 2018

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

Functions to generate stellar source inputs for radmc3d. Warning: check
the `dir_spectra` (l47) to point to the good model directory.

Core functions are: 

`sed_PoWR_models()`:
    Read the SED for the input WR stars parameters (PoWR models, v2018),\n
`sed_kurucz_model()`:
    Read the SED for the input main sequence stars (Kurucz models, use of
    radmc3dPy interface).\n
    
    
------------------------------------------------------------------------
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy import integrate as ip
from scipy.interpolate import interp1d
from termcolor import cprint
from astropy import constants as const
from munch import munchify as dict2class

from asradmc.radmc3dPy import staratm
from asradmc.tools import find_nearest, norm

# Some natural constants

au = 1.49598e13  # Astronomical Unit       [cm]
pc = 3.08572e18  # Parsec                  [cm]
ms = 1.98892e33  # Solar mass              [g]
ts = 5.78e3  # Solar temperature       [K]
ls = 3.8525e33  # Solar luminosity        [erg/s]
rs = 6.96e10  # Solar radius            [cm]

# ================== CHANGE THE MODELS DIRECTORY HERE ==================
dir_spectra = os.environ.get('dir_spectra')
# ======================================================================

if dir_spectra is None:
    cprint("Atmosphere models directory not set:", 'red')
    cprint("-> Add environment variable to your .bash_profile.", 'red')
    cprint("-> e.g.: export dir_spectra='atm_model_directory_path'", 'red')
if (dir_spectra is not None) and not os.path.exists(dir_spectra):
    cprint("Atmosphere models directory not found.", 'red')


def sed_PoWR_models(lam, param, year=2018, display=False, verbose=True):
    """
    Interpolate PoWR stellar atmosphere models for given set of parameters
    (only for carbonaceous star (WC)).

    Input:
    ------
    `lam` {array}: 
        Wavelength domain [µm],\n
    `param` {dict}:  
        Set of stellar parameters: 
            - tstar : effective temperature [K] (optionnal if sptype exist),\n
            - rstar : stellar radius (normalize the luminosity),\n
            - Dinf : terminal radius [:math:`r_\odot`],\n
            - vinf : terminal wind speed [km/s],\n
            - Mdot : mass-loss [:math:`M_\odot/yr`],\n
            - rstar : stellar radius [:math:`r_\odot`],\n
            - dpc : astrometric distance [pc],\n
            - sptype : spectral type (optionnal).
    - year (int):
        Release year of the models (default 2018).

    Returns:
    -------
    - fstar {object} :
        interpolated flux along lam at the given distance (fstar.real) or at 1 pc (fstar.radmc).
        Flux are given in different units : .si [W/m2/:math:`\mu m`], .Jy [Jansky] or .cgs [erg/cm2/s/Hz].
    """
    if year == 2018:
        datadir = dir_spectra + "PoWR/griddl-wc-2018-sed/"
    else:
        datadir = dir_spectra + "PoWR/griddl-wc-sed/"

    ref = np.loadtxt(datadir + "modelparameters.txt", dtype=str, skiprows=8)

    sigma = constants.sigma  # Boltzmann

    tab_Teff = np.array(ref[:, 1], dtype=float)
    R_trans = np.array(ref[:, 2], dtype=float)

    Teff = param["tstar"]
    Rstar = param["rstar"] * rs
    vinf = param["vinf"]
    Dinf = param["Dinf"]
    Mdot = param["Mdot"]
    dpc = param["dpc"] * pc

    try:
        sptype = param["sptype"]
    except KeyError:
        sptype = "Not given"

    Lstar = (4 * np.pi * (Rstar * 1e-2) ** 2 * sigma * Teff ** 4) / (3.8275e26)

    # Transformed radius (Schmutz et al. 1989):
    Rt = (Rstar / rs) * ((vinf / 2500.0) /
                         ((Mdot * np.sqrt(Dinf)) / 1e-4)) ** (2 / 3.0)

    if verbose:
        print("\nPoWR SPTYPE = %s =================" % sptype)
        print("Rstar = %2.1f corresponds to Rt = %2.2f Rsun" % (Rstar / rs, Rt))
    Tmin = np.min(tab_Teff)
    Tmax = np.max(tab_Teff)

    if (Teff < Tmin) or (Teff > Tmax):
        print("------ Warning ------")
        print("PoWR models : %2.0fK < Teff < %2.0fK" % (Tmin, Tmax))
        print("---------------------\n")

    close_T = find_nearest(tab_Teff, Teff)
    close_r = find_nearest(R_trans[(tab_Teff == close_T)], Rt)

    close_model = ref[(tab_Teff == close_T) & (R_trans == close_r)][0]
    good_name = close_model[0]
    if verbose:
        print("Closest model found in PoWR database: Name %s" % close_model[0])
        print("--------------------------------")
        print(
            "Teff = %s K;  Rt = %s Rsun;   M = %s Msun"
            % (close_model[1], close_model[2], close_model[3])
        )
        print(
            "logg = %s cgs;   logL = %s Lsun;   logMdot = %s Msun/yr"
            % (close_model[4], close_model[5], close_model[6])
        )
        print("vinf = %s km/s" % (close_model[7]))

    if year == 2018:
        fname = "wc-2018_%s_sed.txt" % good_name
    else:
        fname = "wc_%s_sed.txt" % good_name

    model = np.loadtxt(datadir + fname)

    wl = 10 ** model[:, 0]  # Angstrom
    wl_micron = wl * 1e-10 * 1e6  # micron
    flux = 10 ** model[:, 1]  # erg/cm2/s/A @ 10pc

    L_model = (
        (4.0 * np.pi * (10 * pc) ** 2) * ip.trapz(flux, wl) / ls
    )  # Luminosity model

    flux_pc = flux * (10.0 / param["dpc"]) ** 2  # flux @ d*pc from users

    flux_scaled_L = (
        Lstar / L_model
    ) * flux_pc  # Normalised luminosity to 4 * pi * R^2 * sig * T^4

    if verbose:
        print("Luminosity scaled L = %2.0f Lsun.\n" % Lstar)

    flux_Jy = fluxToJy(flux_scaled_L, wl_micron / 1e6, 4)
    flux_si = fluxToJy(flux_Jy, wl_micron / 1e6, 1, reverse=True)

    f_planck = norm(Planck_law(Teff, wl_micron / 1e6)) * np.max(flux_si) / 2.5

    flux_cgs = 1e-23 * flux_Jy  # RADMC format [erg/cm2/s/Hz]

    flux_cgs_cl = flux_cgs[flux_si > 1e-50]
    wl_micron_cl = wl_micron[flux_si > 1e-50]

    f_cgs = interp1d(
        wl_micron_cl, flux_cgs_cl, kind="linear", bounds_error=False, fill_value=0
    )

    f_star_cgs = f_cgs(lam)
    f_star_jy = f_star_cgs / 1e-23
    f_star_si = fluxToJy(f_star_jy, lam / 1e6, 1, reverse=True)

    f_star_cgs_1pc = f_star_cgs * param["dpc"] ** 2
    f_star_jy_1pc = f_star_jy * param["dpc"] ** 2
    f_star_si_1pc = f_star_si * param["dpc"] ** 2

    f_star = {
        "real": {"si": f_star_si, "Jy": f_star_jy, "cgs": f_star_cgs},
        "radmc": {"si": f_star_si_1pc, "Jy": f_star_jy_1pc, "cgs": f_star_cgs_1pc},
    }

    fstar = dict2class(f_star)

    if display:
        plt.figure(figsize=(8, 6))
        plt.loglog(lam, f_star_si, linewidth=1,
                   label="PoWR model (%2.0fpc) scaled (logL=%2.2f)"
                   % (dpc / pc, np.log10(Lstar)))
        plt.loglog(wl_micron, f_planck, linewidth=1, linestyle="--",
                   label="BB; Teff=%2.0fK" % Teff)
        plt.ylabel(r"Spectral radiance [$\rm W/m^2/\mu m$]", fontsize=12)
        plt.xlabel(r"Wavelength [$\mu m$]", fontsize=12)
        plt.legend(fontsize=12, loc="best")
        plt.ylim([1e-7 * np.max(flux_scaled_L), 1e1 * np.max(flux_scaled_L)])
        plt.xlim([1e-2, 10])
        plt.grid(which="both", alpha=0.2, color="gray")

    return fstar


def sed_kurucz_model(lam, param, display=False, verbose=True):
    """
    Interpolate Kurucz stellar atmosphere model fora given set of parameters.

    Input:
    ------
    - lam {array like} : 
        Wavelength domain [:math:`\mu m`].
    - param {dict} :  
        Set of stellar parameters.
            - sptype : spectral type (to determine the good teff and logg),
            - tstar : effective temperature [K] (optionnal if sptype exist),
            - logg : surface gravitation (optionnal if sptype exist),
            - rstar : stellar radius [:math:`r_\odot`],
            - dpc : astrometric distance [pc].

    Returns:
    -------
    - fstar {object} :
        interpolated flux along lam at the given distance (fstar.real) or at 1 pc (fstar.radmc).
        Flux are given in different units : .si [W/m2/:math:`\mu m`], .Jy [Jansky] or .cgs [erg/cm2/s/Hz].
    """
    rstar = param["rstar"] * rs
    dpc = param["dpc"] * pc
    sptype = param["sptype"]

    mdir = dir_spectra + "kurucz/"

    # Spectral type to Kurucz model conversion
    conv = np.loadtxt(mdir + "conv_SpType.txt", skiprows=4, dtype=str)
    tab_sptype = conv[:, 0]
    tab_teff = conv[:, 1]
    tab_logg = conv[:, 2]
    tab_name = conv[:, 3]

    if type(sptype) == str:
        if sptype in tab_sptype:
            model_rec = tab_name[tab_sptype == sptype][0]

            teff_rec2 = float(model_rec.split("_")[1].split("[")[0])
            logg_rec2 = (float(model_rec.split("_")[1].split("[")[1][1])
                         + float(model_rec.split("_")
                                 [1].split("[")[1][2]) / 10.0)

            teff_rec = float(tab_teff[tab_sptype == sptype][0])
            logg_rec = float(
                tab_logg[tab_sptype == sptype][0].replace("+", "", 1))

            if verbose:
                print("\nKURUCZ SPTYPE = %s =================" % sptype)
                print("Corresponding Teff = %2.0fK; logg = %2.2f" %
                      (teff_rec, logg_rec))
                print("Recommanded Teff = %2.0fK; logg = %2.2f" %
                      (teff_rec2, logg_rec2))

            teff = teff_rec2
            logg = logg_rec2
        else:
            print("------ Warning ------")
            print("No recommandation for this sptype, try Teff and logg")
            print("---------------------\n")
            try:
                logg = param["logg"]
            except KeyError:
                cprint("Model not found : logg required!", "red")
                return None
            try:
                teff = param["tstar"]
            except KeyError:
                cprint("Model not found : tstar required!", "red")
                return None
    else:
        print("No SPTYPE =============")
        logg = param["logg"]
        teff = param["tstar"]
        print("sptype")

    try:
        b = staratm.getSpectrumKurucz(verbose=verbose, teff=teff, logg=logg, rstar=rstar, wav=lam,
                                      modeldir=mdir)  # erg/s/Hz
    except Exception:
        print("Interpolation of the stellar atmosphere models impossible -> try a SpType")
        print(tab_sptype[2:])
        return None

    lnu = b["lnu"]
    Fnu = lnu / (4 * np.pi * dpc ** 2)  # erg/cm2/s/Hz
    Fnu[Fnu == 0] = 1e-100

    f_cgs = interp1d(b["wav"], np.log10(Fnu), kind="linear", bounds_error=False,
                     fill_value=-50)

    f_star = 10 ** f_cgs(lam)
    f_star_cgs = 10 ** f_cgs(lam)
    f_star_jy = f_star_cgs / 1e-23
    f_star_si = fluxToJy(f_star_jy, lam / 1e6, 1, reverse=True)

    f_star_cgs_1pc = f_star_cgs * param["dpc"] ** 2
    f_star_jy_1pc = f_star_jy * param["dpc"] ** 2
    f_star_si_1pc = f_star_si * param["dpc"] ** 2

    f_star = {"real": {"si": f_star_si, "Jy": f_star_jy, "cgs": f_star_cgs},
              "radmc": {"si": f_star_si_1pc, "Jy": f_star_jy_1pc, "cgs": f_star_cgs_1pc}}

    fstar = dict2class(f_star)

    if display:
        plt.figure(figsize=(8, 6))
        plt.loglog(lam, f_star_cgs, linewidth=1, label="Interpolated SED")
        plt.legend(loc="best")
        plt.xlabel("Wavelength [$\mu m$]")
        plt.ylabel("Spectral irradiance [erg/s/Hz/cm2]")
        plt.ylim(1e-5 * np.max(Fnu), 10 * np.max(Fnu))
        plt.xlim([1e-2, 1e2])
        plt.grid(which="both", alpha=0.2, color="gray")
        plt.tight_layout()
    return fstar


def select_star_model(lam, param, model_type, verbose=True):
    """Select the appropriate model among available (bb, kurucz, PoWR).

    Parameters
    ----------
    `lam` {float}
        Wavelengths [µm],\n
    `param` {dict}
        Dictionnary parameters of the stars,\n
    `model_type` {str}
        Model type (bb, kurucz, PoWR),\n
    `verbose` : {bool}
        Print useful informations, by default True.

    Returns
    -------
    `f_star_cgs` {array}  
        SED of the star.
    """

    f_star_cgs = None
    if model_type == "bb":
        rstar = param["rstar"]
        tstar = param["tstar"]
        f_star_si = np.pi * (rstar ** 2 / (1 * pc) ** 2) * \
            Planck_law(tstar, lam * 1e-6)
        f_star_cgs = 1e-23 * fluxToJy(f_star_si, lam * 1e-6, 1)
    elif model_type == "kurucz":
        f_star = sed_kurucz_model(lam, param, verbose=verbose)
        try:
            f_star_si = f_star.radmc.si
            f_star_cgs = f_star.radmc.cgs
        except Exception:
            f_star_cgs = None
    elif model_type == "PoWR":
        f_star = sed_PoWR_models(lam, param, verbose=verbose)
        try:
            f_star_si = f_star.radmc.si
            f_star_cgs = f_star.radmc.cgs
        except Exception:
            f_star_cgs = None

    return f_star_cgs


def compute_luminosity(sed_star, types, verbose=False):
    """ Compute total luminosity of the input stellar sources. """
    lam = sed_star['lam']
    fstar1 = 1e23 * sed_star["fstar1"]

    sed_star1_si = fluxToJy(fstar1, lam * 1e-6, 1, True)
    f_sed_star1 = interp1d(lam, fstar1)
    f_star1_vis = f_sed_star1(0.5)
    D = 1 * pc
    L1 = (4 * np.pi * (D * 1e-2) ** 2) * \
        ip.trapz(sed_star1_si, lam) / const.L_sun.value
    L2 = ratio_vis = None

    if len(types) == 2:
        fstar2 = 1e23 * sed_star["fstar2"]
        sed_star2_si = fluxToJy(fstar2, lam * 1e-6, 1, True)
        L2 = (4 * np.pi * (D * 1e-2) ** 2) * \
            ip.trapz(sed_star2_si, lam) / const.L_sun.value
        f_sed_star2 = interp1d(lam, fstar2)
        f_star2_vis = f_sed_star2(0.5)
        ratio_vis = f_star2_vis / f_star1_vis

    if len(types) == 1:
        if verbose:
            cprint("\nLtot = %2.1f Lsun" % (L1), color="magenta")
        return L1
    elif len(types) == 2:
        if verbose:
            cprint("\nLtot = %2.1f + %2.1f = %2.1f Lsun" %
                   (L1, L2, L1 + L2), color="magenta",)
            cprint("ratio[0.5mu] = %2.2f" % ratio_vis, color="magenta")
        return L1 + L2


def Planck_law(T, wl, norm=False):
    h = 6.62606957e-34
    c = 299792458.
    k = 1.3806488e-23
    sigma = 5.670373e-8
    P = (4 * np.pi**2) * sigma * T**4

    B = ((2 * h * c**2 * wl**-5) /
         (np.exp(h * c / (wl * k * T)) - 1)) / 1e6  # W/m2/micron
    if norm:
        res = B / P  # kW/m2/sr/m
    else:
        res = B
    return res


def fluxToJy(flux, wl, alpha, reverse=False):
    """
    Convert flux Flambda (in different unit) to spectral
    flux density Fnu in Jansky or the reverse.

    Parameters :
    ----------

    flux {float}:
        Fλ [unit]
    wl {float}:
        Wavelenght [m]
    unit {str}:
        Unit of Fλ (see tab units)
    reverse {boolean}:
        reverse the formulae if True

    Units :
    -----

    Constant conversion depend of Fλ unit :

    =======================   =====   ========
    ``Fλ measured in``        α       β
    =======================   =====   ========
    W/m2/m                    0       3×10-6
    W/m2/μm                   1       3×10–12
    W/cm2/μm                  2       3×10–16
    erg/sec/cm2/μm            3       3×10–9
    erg/sec/cm2/Å             4       3×10–13
    =======================   =====   ========

    References :
    ----------

    [1] Link STSCI http://www.stsci.edu/hst/nicmos/documents/handbooks/current_NEW/Appendix_B.14.3.html
    [2] Wikipedia blackbody https://en.wikipedia.org/wiki/Black_body
    """

    if alpha == 0:
        beta = 3e-6
    elif alpha == 1:
        beta = 3e-12
    elif alpha == 2:
        beta = 3e-16
    elif alpha == 3:
        beta = 3e-9
    elif alpha == 4:
        beta = 3e-13
    else:
        print('Bad unit of flux')
        return None

    wl2 = wl * 1e6  # wl2 in micron
    if reverse:
        out = (flux * beta) / wl2**2
    else:
        out = (flux * wl2**2) / beta
    return out
