# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:13:36 2017

@author: fmillour
"""

import time

import matplotlib
matplotlib.use('TkAgg')

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import interpolation as ip
from termcolor import cprint

from astools.all import AllMyFields

# Some natural constants
au = 1.49598e13     # Astronomical Unit       [cm]
pc = 3.08572e18     # Parsec                  [cm]
ms = 1.98892e33     # Solar mass              [g]
ts = 5.78e3         # Solar temperature       [K]
ls = 3.8525e33      # Solar luminosity        [erg/s]
rs = 6.96e10        # Solar radius            [cm]


def Plot_3D_struc(rhod_dust, m=1e-30, azimuth=-75, save=False):
    """
    Display the 3D structure of the model (3d array like).

    Parameters:
    -----------

    rhod_dust : numpy.array (nx, ny, nz)
        3D array of the density map.
    m : float
        Limit under which the density is set to zero (default m = 1e-30).
    azimuth : float
        Position angle to display the model (default azimuth = -75).
    save : boolean
        True if you want to save the figure as a PDF file (3D_struc_model.pdf).

    Returns:
    --------
    """
    cond = rhod_dust <= m
    rhod_dust2 = rhod_dust.copy()
    rhod_dust2[cond] = 0

    dens_max = rhod_dust2.max()

    cbar_max = np.max(rhod_dust2)
    dens_max = float('%2.0e' % cbar_max)

    cmap = plt.get_cmap("RdYlBu_r")
    norm = LogNorm(dens_max/10., dens_max)
    colors = cmap(norm(rhod_dust2))

    colors[:, :, :, 3] = .5  # Define alpha

    # and plot everything
    cbar_min = np.min(rhod_dust2)
    cbar_max = np.max(rhod_dust2)
    cbarlabels = np.linspace(np.floor(cbar_min), np.ceil(
        cbar_max), num=8, endpoint=True)

    cbarlabels = [float('%2.0e' % cbar_max)]
    for i in [10]:
        cbarlabels.append(dens_max/i)

    fig = plt.figure(figsize=(10, 8))
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.voxels(rhod_dust2, facecolors=colors)  # , vmin = 1e-21, vmax = 1e-19)
    ax1 = fig.add_axes([0.86, 0.2, 0.05, 0.6])
    mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                              norm=norm,
                              orientation='vertical')
    ax.view_init(elev=0, azim=azimuth)
    plt.subplots_adjust(top=1.0,
                        bottom=0.0,
                        left=0.0,
                        right=0.95,
                        hspace=0.0,
                        wspace=0.0)

    if save:
        plt.savefig('3D_struc_model.pdf')
    plt.show(block=False)
    return None


def make_density_pinwheel3D(grid, param):
    """Make the 3-D density array of a spirale"""
    xc = grid[0]
    yc = grid[1]
    zc = grid[2]

    rho0 = param['rho0']
    rin = param['rin']
    rout = param['rout']
    nturn = param['nturn']
    rate = param['rate']
    q_dens = param['q_dens']
    sizex = param['sizex']
    d_choc = param['d_choc']
    dpc = param['dpc']
    spr_factor = param['spr_factor']

    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing='ij',
                             sparse=False, copy=True)

    rx = np.sqrt(xx**2+yy**2)
    ry = np.sqrt(xx**2+zz**2)  # Distance map 2D y coordinates
    rz = np.sqrt(yy**2+zz**2)
    rr = np.sqrt(xx**2+yy**2+zz**2)  # Distance map 3D
    hx = np.sign(xx) * np.abs(xx)  # "vertical" scale
    hy = np.sign(yy) * np.abs(yy)  # "vertical" scale
    hz = np.sign(zz) * np.abs(zz)

    scal = np.cos(np.deg2rad(param['alpha']/2.))
    thet = (rr / rate) * 2 * np.pi

    # Archimedian spiral computation
    yy2 = (hx * np.cos(thet) + hz * np.sin(thet))  # + 100*au
    zz2 = (-hx * np.sin(thet) + hz * np.cos(thet))  # + 100*au
    rx2 = np.sqrt((yy)**2+(1./spr_factor) *
                  (zz2)**2)  # cylinder spiral (1/0.66)

    r3 = rx2 / (rr / sizex)  # conic coordinates along spiral

    rhod = rho0 * (r3 <= rout) * (r3 >= rin) * \
        (yy2 > 0)  # condition to put density

    nn = int(r3.shape[0]/2.)
    nny = int(r3.shape[1]/2.)
    r3_plot = r3.copy()
    r3_plot[(yy2 > 0)] = r3.max()
    mapx = r3_plot[nn, :, :]/au
    mapy = r3_plot[:, nny, :]/au

    if False:
        plt.figure(figsize=(6.5, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(mapx, cmap='terrain')
        cb = plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(mapy, cmap='terrain')
        plt.xticks([])
        plt.yticks([])
        cb = plt.colorbar()
        cb.set_label('Distance [AU]')
        plt.subplots_adjust(top=0.906,
                            bottom=0.009,
                            left=0.0,
                            right=0.947,
                            hspace=0.2,
                            wspace=0.111)

    # If rho depends of r

    rhod2 = rhod / rr**q_dens
    rhod2[rhod2 == 0] = 1e-100

    # Remove dust under sublimation zone
    rhod2[rr <= d_choc*dpc*au] = 1e-100

    # Remove dust after nturn
    rhod2[rr >= nturn*rate] = 1e-100

    rhod_dust = rhod2/np.max(rhod2)*rho0

    cond_spiral = (r3 <= rout) & (r3 >= rin) & (yy2 > 0)
    cond_sub = (rr > d_choc*dpc*au) & (rr < nturn*rate)
    return rhod_dust, cond_spiral, cond_sub, hy, rr


def check_mass(rhod_dust, Mdust, sizex, n):
    m_tot_good = np.sum(rhod_dust*((sizex/n)**3)/ms)
    f = (m_tot_good/Mdust)
    rhod_dust = rhod_dust/f
    return rhod_dust


def put_dust_out_uniform(req_out, rhod_dust, cond_spiral, cond_sub,
                         sizex, n, Mdust, hy, rr, d_choc, display=False):

    nx = rhod_dust.shape[0]
    rnuc = (d_choc*1000*2.58)
    e_disk = 10  # AU
    cond_disk = (hy < e_disk * au) & (hy > -e_disk * au) & (rr > rnuc*au)

    if display:
        plt.figure()
        plt.subplot(121)
        plt.imshow(hy[int(nx/2), :, :], cmap='bwr')
        plt.subplot(122)
        plt.imshow(cond_disk[int(nx/2), :, :])

    dens_in = rhod_dust[cond_spiral]
    dens_out = rhod_dust[~cond_spiral & cond_disk & cond_sub]
    m_in = np.sum(dens_in*((sizex/n)**3)/ms)
    m_out = np.sum(dens_out*((sizex/n)**3)/ms)

    m_out_req = (req_out/100.) * Mdust
    m_in_req = Mdust - m_out_req

    dens_out2 = dens_out * (m_out_req/m_out)
    dens_in2 = dens_in * (m_in_req/m_in)

    m_out2 = np.sum(dens_out2*((sizex/n)**3)/ms)
    m_in2 = np.sum(dens_in2*((sizex/n)**3)/ms)

    rhod_dust[cond_spiral] = dens_in2
    rhod_dust[~cond_spiral & cond_disk & cond_sub] = dens_out2
    m_tot_good2 = np.sum(rhod_dust*((sizex/n)**3)/ms)
    rel_in_out = 100*(m_out2/m_tot_good2)

    print('\nDust mass configuration:')
    print('-------------------------')
    print('M_spiral = %2.2e Msun, M_env = %2.2e Msun (%2.2f %%)' %
          (m_in2, m_out2, rel_in_out))

    return rhod_dust, m_tot_good2


def make_spiral_GRID(pinwheel_param, n=64, fov=0.3, ratio_y=1, posang=0, display=False):
    """ Save the density grid files for radmc"""
    start_time = time.time()
    # __________________________________________________________________________
    # Physical parameters
    # __________________________________________________________________________
    cprint('=> Compute dust 3D density map...', color='green')
    cprint('---------------------------------', color='green')

    dpc = pinwheel_param['dpc']
    nturn = pinwheel_param['nturn']

    p_escape = pinwheel_param['p_escape']
    # mdot  = pinwheel_param['mdot']
    # P     = pinwheel_param['P']
    # xi    = pinwheel_param['xi']/100.

    spr_factor = pinwheel_param['spr_factor']

    Mdust = pinwheel_param['Mdust']  # nturn * mdot * (P/365.25) * xi
    # Mdot_dust = Mdust / nturn / (P/365.25)
    # cprint(('Total mass dust in %2.2f turns with xi = %2.2f %% : mdust = %2.2e Msun'%(nturn, xi*100, Mdust)), color = 'magenta')
    # cprint(('=> Dust formation rate : Mdot_d = %2.2e Msun/yr\n'%(Mdot_dust)), color = 'magenta')

    # __________________________________________________________________________
    # Grid parameters
    # __________________________________________________________________________

    # n        = npix    #Number of elements into the grid
    nx = n  # Number of elements into the grid in x direction
    ny = n/ratio_y  # Number of elements into the grid in y direction
    nz = n  # Number of elements into the grid in z direction

    sizex = fov*(dpc*au)  # fov in x [as]
    sizey = (fov/ratio_y)*(dpc*au)  # fov in y [as]
    sizez = fov*(dpc*au)  # fov in z [as]

    print(("Size %d -> N cells = %2d" % (n, n**3/ratio_y)))
    # ______________________________________________________________________________
    # Model parameters
    # ______________________________________________________________________________

    opening_angle = pinwheel_param['alpha']
    omega = pinwheel_param['omega']
    d_choc = pinwheel_param['rnuc']

    q_dens = pinwheel_param['q_dens']
    r_out_open = np.sin(np.deg2rad(opening_angle/2.))*sizex/(dpc*au)
    r_in_open = (1-omega)*r_out_open

    rho0 = 1  # Density in dust creation location [g/cm3]
    rin = r_in_open*(dpc*au)  # Intern radius of the spiral [as]
    rout = r_out_open*(dpc*au)  # Outer radius of the spiral [as]
    rate = pinwheel_param['step']*(dpc*au)/1000.

    # ______________________________________________________________________________
    # Make the coordinates
    # ______________________________________________________________________________

    nx, ny, nz = int(nx), int(ny), int(nz)

    xi = np.linspace(-sizex/2., sizex/2., nx+1)
    yi = np.linspace(-sizey/2., sizey/2., ny+1)
    zi = np.linspace(-sizez/2., sizez/2., nz+1)

    xc = (xi[0:nx] + xi[1:nx+1])//2  # Centering to the middle
    yc = (yi[0:ny] + yi[1:ny+1])//2
    zc = (zi[0:nz] + zi[1:nz+1])//2

    # ______________________________________________________________________________
    # Make the dust density model
    # ______________________________________________________________________________

    arbit_rho0 = 1  # Arbitrary density in dust creation location [g/cm3]

    arbit_param = {'rin': rin,
                   'rout': rout,
                   'rho0': rho0,
                   'nturn': nturn,
                   'rate': rate,
                   'q_dens': q_dens,
                   'sizex': sizex,
                   'd_choc': d_choc,
                   'dpc': dpc,
                   'n': n,
                   'spr_factor': spr_factor,
                   'alpha': opening_angle
                   }

    arbit_rhod_dust, cond_spiral, cond_sub, hy, rr = make_density_pinwheel3D(
        [xc, yc, zc], arbit_param)

    # Compute density map with arbitrary dust density at d_choc
    arbit_m_tot = np.sum(arbit_rhod_dust*((sizex/n)**3)/ms)

    # Compute the dust density to get the good total mass (cross product)
    new_rho0 = Mdust * arbit_rho0/arbit_m_tot

    param = arbit_param.copy()
    param['rho0'] = new_rho0

    rhod_dust, cond_spiral, cond_sub, hy, rr = make_density_pinwheel3D(
        [xc, yc, zc], param)

    rhod_dust = check_mass(rhod_dust, Mdust, sizex, n)

    rhod_dust, final_mdust = put_dust_out_uniform(p_escape, rhod_dust, cond_spiral, cond_sub,
                                                  sizex, n, Mdust, hy, rr, d_choc)
    param['Mtot'] = final_mdust
    print(('CHECK : Computed total mass Mdust = %2.2e Msun\n' % final_mdust))

    scale = 'mas'
    if scale == 'AU':
        s = rate/au
        extentx = np.array(
            [-sizez/2./au, sizez/2./au, -sizey/2./au, sizey/2./au])
        extenty = np.array(
            [-sizex/2./au, sizex/2./au, -sizez/2./au, sizez/2./au])
        extentz = np.array(
            [-sizey/2./au, sizey/2./au, -sizex/2./au, sizex/2./au])
    elif scale == 'cm':
        s = rate
        extentx = np.array([-sizez/2., sizez/2., -sizey/2., sizey/2.])
        extenty = np.array([-sizex/2., sizex/2., -sizez/2., sizez/2.])
        extentz = np.array([-sizey/2., sizey/2., -sizex/2., sizex/2.])
    else:
        s = 1000*rate/au/dpc
        extentx = 1e3*np.array([-sizez/2./au, sizez /
                                2./au, -sizey/2./au, sizey/2./au])/dpc
        extenty = 1e3*np.array([-sizex/2./au, sizex /
                                2./au, -sizez/2./au, sizez/2./au])/dpc
        extentz = 1e3*np.array([-sizey/2./au, sizey /
                                2./au, -sizex/2./au, sizex/2./au])/dpc

    xlim = s*np.cos(np.linspace(0, 2*np.pi, 100))
    ylim = s*np.sin(np.linspace(0, 2*np.pi, 100))

    if display:
        mid_plan_im = rhod_dust[:, int(ny)//2+1, :]
        mid_plan_im_rotated = ip.rotate(mid_plan_im, 90+posang)

        born = np.max(extentx)
        yborn = np.tan(np.deg2rad(opening_angle/2.))*born

        plt.figure(figsize=(6, 5))
        plt.subplot(2, 2, 1)
        plt.text(0.2*np.max(extentx), .75*np.max(extentx),
                 'Projected', color='w', fontsize=12)
        plt.imshow(np.sum(rhod_dust, axis=0), norm=PowerNorm(.5),
                   extent=extentx, origin='lower', cmap='gist_earth')
        for i in range(int(nturn)):
            plt.plot((i+1)*xlim, (i+1)*ylim, lw=1, color='#f41ea1')

        plt.plot([0, -born], [0, yborn], '--', lw=1, color='#50bf92')
        plt.plot([0, -born], [0, -yborn], '--', lw=1, color='#50bf92')
        plt.plot([0, born], [0, yborn], '--', lw=1, color='#50bf92')
        plt.plot([0, born], [0, -yborn], '--', lw=1, color='#50bf92')
        plt.ylabel('r [%s]' % scale, fontsize=12)
        plt.xticks([])
        plt.plot(0, 0, 'wo', alpha=.5)
        plt.subplot(2, 2, 2)
        plt.plot(0, 0, 'wo', alpha=.5)
        for i in range(int(nturn)):
            plt.plot((i+1)*xlim, (i+1)*ylim, lw=1, color='#f41ea1')
        plt.imshow(np.sum(rhod_dust, axis=1), norm=PowerNorm(.5),
                   extent=extenty, origin='lower', cmap='gist_earth')
        # plt.plot(xlim, ylim, color='#f41ea1')
        plt.xticks([])
        plt.yticks([])

        sumZ = np.sum(rhod_dust, axis=2)
        sumZ[sumZ <= 0] = 0

        # plt.subplot(2, 3, 3)
        # # plt.title('Z map')
        # # plt.title('Summed Density Z axis')
        # plt.plot(0, 0, 'wo', alpha=.5)
        # plt.imshow(sumZ, extent=extentz,
        #            norm=PowerNorm(0.5), origin='lower')

        # cb = plt.colorbar()
        # # plt.xlabel('r [%s]'%scale)
        # cb.set_label(r'$\rho$ [g/cm$^3$]')
        # plt.xticks([])
        # plt.yticks([])
        plt.subplot(2, 2, 3)
        # plt.title('Slice Density X axis')
        plt.imshow(rhod_dust[int(nx)//2+1, :, :],
                   norm=PowerNorm(.5), extent=extentx, origin='lower', cmap='gist_earth')
        plt.text(.5*np.max(extentx), .75*np.max(extentx),
                 'Slice', color='w', fontsize=12)
        plt.xlabel('r [%s]' % scale, fontsize=12)
        plt.ylabel('r [%s]' % scale, fontsize=12)
        plt.plot(0, 0, 'wo')
        plt.subplot(2, 2, 4)
        # plt.title('Slice Density Y axis')
        plt.plot(0, 0, 'wo')
        plt.imshow(rhod_dust[:, int(ny)//2+1, :],
                   norm=PowerNorm(.5), extent=extenty, origin='lower', cmap='gist_earth')
        plt.yticks([])
        plt.xlabel('r [%s]' % scale, fontsize=12)
        # plt.subplot(2, 3, 6)
        # # plt.title('Slice Density Z axis')
        # plt.plot(0, 0, 'wo')
        # plt.yticks([])
        # plt.imshow(rhod_dust[:, :, int(nz)//2+1],
        #            norm=PowerNorm(.5), extent=extentz, origin='lower')
        # plt.xlabel('r [%s]' % scale, fontsize=12)
        # cb = plt.colorbar()
        # cb.set_label(r'$\rho$ [g/cm$^3$]')
        plt.subplots_adjust(top=0.97,
                            bottom=0.095,
                            left=0.07,
                            right=0.97,
                            hspace=0.1,
                            wspace=0.005)
        plt.show(block=False)

    gridinfo = {'nx': nx,
                'ny': ny,
                'nz': nz,
                'xi': xi,
                'yi': yi,
                'zi': zi}

    t = time.time() - start_time
    print(("==> Create dust density map execution time = --- %2.1f s ---\n" % t))
    return rhod_dust, AllMyFields(gridinfo)
