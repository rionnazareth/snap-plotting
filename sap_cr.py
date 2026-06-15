import numpy as np
import astropy.units as u
import astropy.constants as const
import pylab

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.colors as mpl_c
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


import sys
import os
sys.path.insert(0, '/cosma8/data/dp317/dc-naza3/arepo-snap-util')
import arepo_run as arun
import gadget
import scienceplots
from tests.lib import *

plt.style.use(['science'])

if __name__ == "__main__":
    print('Running snap plotting script...')
    slurm_ntasks = os.getenv('SLURM_NTASKS', '').strip()
    numthreads = int(slurm_ntasks) if slurm_ntasks.isdigit() and int(slurm_ntasks) > 0 else 1
    BASE_PATH = '/cosma8/data/dp317/dc-naza3/homogeneous'
    SNAPBASE = 'snap_'
    SNAPFILETYPE = '.hdf5'

    GAMMA = 5/3 # heat capaticy ratio for monatomic gas
    HYDROGENMASS_FRAC = 0.76

    # k_B = const.k_B.to((u.Msun*1e10)*(u.km/u.s)**2/u.K).value
    # m_p = const.m_p.to(u.Msun*1e10).value

    k_B       = 1.381e-16
    m_p       = 1.66e-24



    ## Set up figure & axis grid
    fig = plt.figure(figsize=(8,8))

    outer = gridspec.GridSpec(1, 1, wspace=0.2)
    quad_subs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0, wspace=0)





    ## Set global figure options

    image_proj = 'side'      # side or top viewing angle; top view shows azimuthal curvature
    plotsize =  0.5  # Size in kpc of one panel
    proj_on = False        # Whether to do a slice or a projection
    proj_fact = 0.1         # Fraction of plotsize to project through
    res = 1024               # Pixels per panel

    ## Load the snapshot
    v = 'xcr'
    c = 'vanimo'
    r = [1e-2,1e2]
    snap_num = 1
    
    add_vec = False
    logplot = True

    

    norm = False
    cdis = False
    def norm_by_snap0(norm):
        s0 = load_snap_data(0,snappath=output_path,snapbase=SNAPBASE)
        if norm:
            for data_key in ['rho', 'nH_cm', 'pres', 'speed','temp']:#
                # Avoid division by zero and cast to float to avoid UFuncTypeError
                div = np.median(s0.data[data_key]) if s0.data[data_key].mean() != 0 else 1e-10
                if data_key == 'cren': div = np.median(s0.data['u'])  # Normalize CR energy by initial internal energy, not CR energy:
                if data_key == 'crpres': div = np.median(s0.data['pres'])  # Normalize CR pressure by initial thermal pressure, not CR pressure
                if data_key == 'speed': div = 1e5/unit_v # speed in kms
                s.data[data_key] = s.data[data_key].astype(float) / div
                print(f'Normalized {data_key} by dividing by max value from snap 0: {np.median(s0.data[data_key])}')
        if cdis: 
            s.data[v] *= (s.data['wind']>=0.5)


    ## Plot each axis quadrent
    # Top left
    output_path = BASE_PATH + '/rhov_hires/5/'

    s = load_snap_data(snap_num,snappath=output_path,snapbase=SNAPBASE)
    snap_time_TL = calc_snap_time(s)

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3
    
    # Here we're passing the same snap each time, but you could give each one a different snapshot to make e.g. a time series image
    norm_by_snap0(norm)
    quad_TL, map_TL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    # output_path = BASE_PATH + '/new/output_cbf/'

    s = load_snap_data(4,snappath=output_path,snapbase=SNAPBASE)
    snap_time_BL = calc_snap_time(s)
    # Bottom left
    norm_by_snap0(norm)
    quad_BL, map_BL = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,0],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    # snap_num = 1
    # output_path = BASE_PATH + '/new/output_cbcr/'

    s = load_snap_data(7,snappath=output_path,snapbase=SNAPBASE)
    snap_time_BR = calc_snap_time(s)
    # Bottom right
    norm_by_snap0(norm)
    quad_BR, map_BR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [1,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )

    # Top right
    # output_path = BASE_PATH + '/output_cturb/'

    s = load_snap_data(9,snappath=output_path,snapbase=SNAPBASE)
    snap_time_TR = calc_snap_time(s)
    norm_by_snap0(norm)
    quad_TR, map_TR = plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,1],
        var = v,
        weighted = 'rho', # or None
        ranges = r,
        cmap = c,
        logplot = logplot,
        divzero = False,
        divzero_centre = None,
        colorbar=False,
        image_proj = image_proj,
        proj_on = proj_on,
        proj_fact = proj_fact,
        res = res,
        numthreads = numthreads,
        plotsize = plotsize,
        add_vec=add_vec,
        vec_val='bfld'
        )



    # Adjust figure to make room for single right-side colorbar
    fig.subplots_adjust(right=0.88)

    # Single shared colorbar on the right
    cax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(map_TL, cax=cax, orientation='vertical', label=r'$\varepsilon_\mathrm{CR}$')

    # Time labels in each panel corner
    quad_TL.text(0.05, 0.95, f'$t = {snap_time_TL:.1f}$ Myr', transform=quad_TL.transAxes,
                 color='white', ha='left', va='top', weight='bold', fontsize=11)
    quad_BL.text(0.05, 0.05, f'$t = {snap_time_BL:.1f}$ Myr', transform=quad_BL.transAxes,
                 color='white', ha='left', va='bottom', weight='bold', fontsize=11)
    quad_TR.text(0.95, 0.95, f'$t = {snap_time_TR:.1f}$ Myr', transform=quad_TR.transAxes,
                 color='white', ha='right', va='top', weight='bold', fontsize=11)
    quad_BR.text(0.95, 0.05, f'$t = {snap_time_BR:.1f}$ Myr', transform=quad_BR.transAxes,
                 color='white', ha='right', va='bottom', weight='bold', fontsize=11)
    fig.suptitle(r'CR Energy Density $\varepsilon_\mathrm{CR}$', fontsize=12)
    fig.savefig('/cosma8/data/dp317/dc-naza3/snap-plotting/tests/crph/{}_snap{}_{}.png'.format(v,number_string(snap_num),image_proj),dpi=300)
    # plt.show()