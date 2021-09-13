#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:29:02 2018

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

General tools for asradmc.
------------------------------------------------------------------------
"""

import numpy as np
from astropy.io import fits
import datetime


def rad2mas(rad):
    """Convert angle in radians to milli arc-sec."""
    mas = rad * (3600.0 * 180 / np.pi) * 10.0 ** 3
    return mas


def mas2rad(mas):
    """Convert angle in milli arc-sec to radians."""
    rad = mas * (10 ** (-3)) / (3600 * 180 / np.pi)
    return rad


def write_fits(file, image, wl, pixel_size, param=None, delta_wl=0, crpix1=0, crpix2=0,
               crval1=0, crval2=0, align=1, Jy=True):

    # Date of file creation
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create header
    hdr = fits.Header()
    if type(wl) == list:
        hdr["NAXIS3"] = len(wl)
    else:
        hdr["NAXIS3"] = 1

    try:
        hdr["CDELT3"] = (np.diff(wl)[0]/1e6, " Spectral step")
        hdr["CRVAL3"] = (wl[0]/1e6, " Wavelenght reference")
        hdr["CRPIX3"] = (1, "Image reference")
        hdr["CUNIT3"] = ("m", "unit wavelenght")
        pass
    except (KeyError, IndexError):
        print('No chromatic model.')
        pass

    hdr["EXTNAME"] = ("IMAGE", "Name of HDU")
    hdr["DATE"] = (date, "file creation date (YYYY-MM-DDThh:mm:ss UT)")
    hdr["CTYPE1"] = (
        "RA---TAN", "first parameter RA  ,  projection TANgential")
    hdr["CTYPE2"] = (
        "DEC--TAN", "second parameter DEC,  projection TANgential")
    # 270.5167 (WR104)
    hdr["CRVAL1"] = (crval1, "Coordinate value of reference pixel")
    # -23.6283 (WR104)
    hdr["CRVAL2"] = (crval2, "Coordinate value of reference pixel")
    try:
        hdr["CRPIX1"] = (param["pstar"][1], " X ref pixel (R.A.)")
        hdr["CRPIX2"] = (param["pstar"][0], " Y ref pixel (dec) ")
    except (KeyError, TypeError):
        hdr["CRPIX1"] = (crpix1, " X ref pixel (R.A.)")
        hdr["CRPIX2"] = (crpix2, " Y ref pixel (dec) ")

    hdr["CDELT1"] = (align*mas2rad(pixel_size), "")
    hdr["CDELT2"] = (mas2rad(pixel_size), "")
    hdr["CUNIT1"] = ('rad     ')
    hdr["CUNIT2"] = ('rad     ')
    hdr["CD1_1"] = (align*np.rad2deg(mas2rad(pixel_size)),
                    " X increment (deg) ")
    hdr["CD2_2"] = (np.rad2deg(mas2rad(pixel_size)), " Y increment (deg) ")
    if Jy:
        hdr["BUNIT"] = ("JY/PIXEL", "Intensity unit of the image")
    else:
        hdr["BUNIT"] = ("Counts/PIXEL", "Intensity unit of the image")

    hdr["BTYPE"] = ("INTENSITY", "Type of the images")
    if len(wl) >= 10.0:
        hdr["WAVE1"] = (str(wl[:-5]), " Wavelenght        ")
        hdr["WAVE2"] = (str(wl[-5:]), " Wavelenght        ")
    else:
        hdr["WAVE1"] = (str(wl), " Wavelenght        ")
    hdr[
        "COMMENT"
    ] = "FITS (Flexible Image Transport System) format is defined in\
        'Astronom and Astrophysics,' volume 376, page 359; bibcode: 2001A&A...376..359H"

    if len(image.shape) == 3:
        fluxes = []
        for im in image:
            fluxes.append(im.sum())
        fluxes = np.array(fluxes)
        hdr["FLUXES"] = (str(fluxes), " Spectrum [Jy]        ")
    # for p in param.keys():
    #     try:
    #         hdr[p] = param[p]
    #     except Exception:
    #         pass
    fits.writeto(file, image, header=hdr, overwrite=True)
    return None


def find_nearest(tab, value):
    idx = (np.abs(tab - value)).argmin()
    v = tab[idx]
    return v


def norm(tab):
    """Normalize the tab array by the maximum."""
    return tab / np.max(tab)
