# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:31:43 2019

@author: Anthony Soulain (University of Sydney)

------------------------------------------------------------------------
asradmc: Tools to perform RT modeling on analytical grid or ramses data
------------------------------------------------------------------------

Function to deal with amr method (adaptive mesh refinement).

------------------------------------------------------------------------
"""

import pickle
import time

import numpy as np
import pandas
from astropy import constants as const
from munch import munchify as dict2class
from termcolor import cprint
from tqdm import tqdm

from asradmc.inputs import make_amr_grid_octree, make_density_file

# Some natural constants
au = const.au.cgs.value  # Astronomical Unit       [cm]


def check_requirement(name_dust_type, density, taumax=1):
    """ Check the requested amr level required to obtain a
    fully optically thin (tau < 1) grid. """
    dustfile = 'dustkappa_%s.inp' % (name_dust_type)

    opac = np.loadtxt(dustfile, skiprows=3)
    kabs = opac[:, 2]
    ksca = opac[:, 1]
    kappa_max = np.max(ksca + kabs)

    # Open grid and density file
    # ------------------------------------------------------------------------------
    grid = np.array(pandas.read_csv("amr_grid.inp", header=None))[:, 0]

    # file = open("dust_density_grid.dpy", "rb")
    # density = pickle.load(file)
    # file.close()

    n_cell = len(density)

    l_grid = grid[6:-1].astype(float)  # Wall of the simulation

    max_grid = np.max(l_grid)
    min_grid = np.min(l_grid)

    nx = int(grid[5].split()[0])  # number of cells in x direction
    ny = int(grid[5].split()[1])  # number of cells in y direction
    nz = int(grid[5].split()[2])  # number of cells in z direction

    size_cell = (max_grid + abs(min_grid)) / nx  # Width of one cell [cm]

    # Optical depth condition (adaptative mesh refinement)
    # ------------------------------------------------------------------------------

    tau_raf1 = density * kappa_max * size_cell / (8.0 ** 0)
    tau_raf2 = density * kappa_max * size_cell / (8.0 ** 1)
    tau_raf3 = density * kappa_max * size_cell / (8.0 ** 2)
    tau_raf4 = density * kappa_max * size_cell / (8.0 ** 3)
    tau_raf5 = density * kappa_max * size_cell / (8.0 ** 4)
    tau_raf6 = density * kappa_max * size_cell / (8.0 ** 5)

    cond_tau_raf1 = tau_raf1 >= taumax
    cond_tau_raf2 = tau_raf2 >= taumax
    cond_tau_raf3 = tau_raf3 >= taumax
    cond_tau_raf4 = tau_raf4 >= taumax
    cond_tau_raf5 = tau_raf5 >= taumax
    cond_tau_raf6 = tau_raf6 >= taumax

    cond_tau = {'r1': cond_tau_raf1,
                'r2': cond_tau_raf2,
                'r3': cond_tau_raf3,
                'r4': cond_tau_raf4,
                'r5': cond_tau_raf5,
                'r6': cond_tau_raf6,
                }

    n_lvl1 = len(density[cond_tau_raf1])
    n_lvl2 = len(density[cond_tau_raf2])
    n_lvl3 = len(density[cond_tau_raf3])
    n_lvl4 = len(density[cond_tau_raf4])
    n_lvl5 = len(density[cond_tau_raf5])
    n_lvl6 = len(density[cond_tau_raf6])

    l_lvl = np.array([n_lvl1, n_lvl2, n_lvl3, n_lvl4, n_lvl5, n_lvl6])
    lvl_required = np.min(np.where(l_lvl == l_lvl.min()))

    if lvl_required == 0:
        print("\nWarning : AMR not required for this model")
    else:
        print("\nLvl AMR required N = %i :" % lvl_required)
        if lvl_required > 0:
            print("-> First order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl1, 100.0 * n_lvl1 / n_cell))
        if lvl_required > 1:
            print("-> Second order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl2, 100.0 * n_lvl2 / n_lvl1))
        if lvl_required > 2:
            print("-> Third order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl3, 100.0 * n_lvl3 / n_lvl2))
        if lvl_required > 3:
            print("-> 4th order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl4, 100.0 * n_lvl4 / n_lvl3))
        if lvl_required > 4:
            print("-> 5th order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl5, 100.0 * n_lvl5 / n_lvl4))
        if lvl_required > 5:
            print("-> 6th order amr needed : %d cells (%2.2f percent)"
                  % (n_lvl6, 100.0 * n_lvl6 / n_lvl5))

    tab_amr = dict2class({'lvl_required': lvl_required, 'nx': nx, 'ny': ny, 'nz': nz,
                          'density': density, 'n_cell': n_cell,
                          'cond_tau': cond_tau
                          })
    return tab_amr


def grid_refinement(t_amr, fov, dpc, ratio_y=1, binary=False):
    """ Perform amr grid refinement using the pre-computed condition
    with check_amr_requirement(). """
    start_time = time.time()
    cprint("=> AMR grid refinement process...", color="green")
    cprint("---------------------------------", color="green")

    # No refined octree (zeros*ncells)
    octree = np.zeros(int(t_amr.n_cell)).astype(int)

    # Assign different value according the required amr level
    # -------------------------------------------------------
    octree[t_amr.cond_tau.r1] = 1
    octree[t_amr.cond_tau.r2] = 2
    octree[t_amr.cond_tau.r3] = 3
    octree[t_amr.cond_tau.r4] = 4
    octree[t_amr.cond_tau.r5] = 5
    octree[t_amr.cond_tau.r6] = 6

    # Filling value for the refinement levels
    # ---------------------------------------
    add_raf0 = [0]  # cell not refined
    add_raf1 = [1] + [0] * 8
    add_raf2 = [1] + 8 * ([1] + [0] * 8)
    add_raf3 = [1] + 8 * ([1] + 8 * ([1] + [0] * 8))
    add_raf4 = [1] + 8 * ([1] + 8 * ([1] + 8 * ([1] + [0] * 8)))
    add_raf5 = [1] + 8 * ([1] + 8 * ([1] + 8 * ([1] + 8 * ([1] + [0] * 8))))

    dic_raf = {"0": add_raf0, "1": add_raf1, "2": add_raf2, "3": add_raf3,
               "4": add_raf4, "5": add_raf5, "6": add_raf5}

    # Compute the final octree (tree of refinement filled with 0 and 1)
    # -----------------------------------------------------------------
    final_octree = []
    for lvl in tqdm(octree, desc='Create octree array', ncols=100, leave=False):
        final_octree.extend(dic_raf[str(lvl)])
    f_octree = np.array(final_octree)

    # Compute the density in the refined cells (if lvl = 1, density*8**1)
    # -------------------------------------------------------------------
    final_density = []
    for i in tqdm(range(len(octree)), desc='Create density array', ncols=100, leave=False):
        final_density.extend([t_amr.density[i]] * 8 ** octree[i])
    final_density_scaled = np.array(final_density)

    # Save the density grid into the new dust_density.inp
    # ---------------------------------------------------
    make_density_file(final_density_scaled, binary=binary)

    # Save the new refined grid into amr_grid.inp
    # -------------------------------------------
    sizex = fov * (dpc * au)  # fov in x [as]
    sizey = (fov / ratio_y) * (dpc * au)  # fov in y [as]
    sizez = fov * (dpc * au)  # fov in z [as]
    xi = np.linspace(-sizex / 2.0, sizex / 2.0, t_amr.nx + 1)
    yi = np.linspace(-sizey / 2.0, sizey / 2.0, t_amr.ny + 1)
    zi = np.linspace(-sizez / 2.0, sizez / 2.0, t_amr.nz + 1)
    nleaf = len(final_density)
    nbranch = len(final_octree)

    make_amr_grid_octree(t_amr.nx, t_amr.ny, t_amr.nz, xi, yi, zi, levelmax=t_amr.lvl_required,
                         nleaf=nleaf, nbranch=nbranch, octree=f_octree)
    t = time.time() - start_time
    print(("\n==> Refinement execution time = --- %2.1f s ---" % t))
    return None
