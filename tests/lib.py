import h5py
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

GAMMA      = 5./3
GAMMA_CR   = 4./3
k_B        = 1.381e-16
m_p        = 1.66e-24
HYDROGENMASS_FRAC = 0.76

import sys, os
sys.path.insert(0, '/home/dc-naza3/arepo-snap-util')
import arepo_run as arun
import gadget

def _count_snaps(path):
    n = 0
    while os.path.exists(os.path.join(path, f'snap_{n:03d}.hdf5')):
        n += 1
    return n

def number_string(snap):
    snap_str = str(snap)
    if len(snap_str) < 3:
        snap_str = (3-len(snap_str))*str(0)+snap_str
    return snap_str

def calc_snap_time(s):
    Length_unit = s.header['UnitLength_in_cm']
    Velocity_unit = s.header['UnitVelocity_in_cm_per_s'] 
    Time_unit = Length_unit / Velocity_unit
    Time = s.header['Time'] * Time_unit
    Time_Myr = (Time * u.s).to(u.Myr).value
    return Time_Myr


def get_vranges(snap_nums,snappath=None,quant='temp',flag_excl=True):
    minval = np.empty(0)
    maxval = np.empty(0)

    for i_s, num in enumerate(snap_nums):
        s = load_snap_data(num,snappath=output_path,snapbase=SNAPBASE)
        quant_data = s.data[quant]
        if flag_excl: #Exclude the BOLA cells
            quant_data = quant_data[s.data['flag']==0]
        minval = np.append(minval, quant_data.min())
        maxval = np.append(maxval, quant_data.max())
        #print('\n',s.data[quant].min(),s.data[quant].max())

    ranges = [minval.min(),maxval.max()]
    print('Min/max values at snaps:',minval.argmin(),minval.argmax())
    return ranges


def calc_mu(data,ionisation='ionised'):
    X = HYDROGENMASS_FRAC

    if 'metals' in data.keys():
        Z = data['metals'] ## metal mass fraction
    else:
        Z = 0

    if 'ne' in data.keys():
        ne = data['ne']
        print('\t>>Electrion density found, calculating mu...')
    elif ionisation == 'ionised':
        ne = np.ones_like(data['u']) * 2 / (1+X)
        print('\t>>Electron density not found, asssuming default ionisation: {}'.format(ionisation))
    elif ionisation =='neutral':
        ne = np.zeros_like(data['u'])
        print('\t>>Electron density not found, asssuming default ionisation: {}'.format(ionisation))

    mu = 4 / (X * (3 + 4*ne) + 1 - Z)
    return mu

def calc_T(data, unit_v):
    temp = data['u']*(unit_v)**2 * (GAMMA - 1) * data['mu']  * m_p  / k_B
    print(temp)
    return temp

def calc_P(data):
    pressure = (data['rho']) * (data['u']) * (GAMMA - 1)
    print('pressure:', pressure)
    return pressure

def calc_CRP(data, gamma_cr=4/3):
    pressure = (data['rho']) * (data['cren']) * (gamma_cr - 1)
    print('CR pressure:', pressure)
    return pressure

def rho_to_n_cm(data, unit_rho):
    n_cms = data['rho'] * unit_rho/ (m_p )
    return n_cms

def rho_to_nH_cm(data, unit_rho):
    nH_cm = HYDROGENMASS_FRAC * data['rho'] * unit_rho/ (m_p )
    return nH_cm


def calc_cooling(data, header, verbose=False):
    if 'coor' in data:
        ## Setup unit system
        length_unit = header['UnitLength_in_cm'] * u.cm
        velocity_unit = header['UnitVelocity_in_cm_per_s'] * u.cm / u.s
        mass_unit = header['UnitMass_in_g'] * u.g
        time_unit = length_unit / velocity_unit

        ## s.data['coor'] is the cooling rate per unit mass, in simulations units

        cool_time =  (time_unit * data['u'] / data['coor']).to(u.Myr)  ## Comes from io_fields.c
        cool_rate_mass = (data['coor'] * velocity_unit**3 / length_unit).to(u.erg / u.s / u.g)

        ratefact = (HYDROGENMASS_FRAC / (m_p * mass_unit))**2 * data['rho'] * mass_unit / length_unit**3
        cool_rate_volume = (cool_rate_mass/ratefact).to(u.erg/u.s * u.cm**3)

        return cool_time.value, cool_rate_mass.value, cool_rate_volume.value
    
    else:
        if verbose:
            print('\t>>Field COOR not found, skipping cooling calculation...')
        return None, None, None
    

def calc_bremsstrahlung(data,T_cut=None,gaunt_factor=1, Z_factor = 1):
    # See Sijacki & Springel 06, also Bennett
    # Bolometric X-ray (all bands)

    xray = np.zeros_like(data['temp'])

    vols = (data['vol']*u.kpc**3).to(u.cm**3).value
    xray =  1.4e-27 * gaunt_factor * Z_factor**2 * data['temp']**(0.5) * data['ne'] * data['nH_cm']**2 * (1-data['nh']) * vols

    if T_cut is not None:
        xray[data['temp']<T_cut] = 0

    return xray


def load_snap_data(num,snappath=None,snapbase='snap_',advanced_xrays=False,verbose=False,default_ionisation='ionised'):
    
    snapname = snappath + snapbase + number_string(num)
    print('\n>Loading snap',snapname)

    o = arun.Run(snappath=snappath,snapbase=snapbase)
    s = o.loadSnap(snapnum=num)

    unit_v = s.header['UnitVelocity_in_cm_per_s']
    unit_l = s.header['UnitLength_in_cm'] 
    unit_m = s.header['UnitMass_in_g']
    unit_t = unit_l / unit_v
    unit_rho = unit_m / unit_l**3

    s.data['mu'] = calc_mu(s.data,ionisation=default_ionisation)
    s.data['temp'] = calc_T(s.data, unit_v=unit_v)

    if 'pres' not in s.data:
        s.data['pres'] = calc_P(s.data)

    if 'wind' not in s.data and 'pass' in s.data:
        s.data['wind'] = s.data['pass']

    s.data['n_dens_cm'] = rho_to_n_cm(s.data, unit_rho=unit_rho)

    s.data['nH_cm'] = rho_to_nH_cm(s.data, unit_rho=unit_rho)
    xx = s.data['pos'][:,0] - s.boxsize/2.
    yy = s.data['pos'][:,1] - s.boxsize/2.
    zz = s.data['pos'][:,2] - s.boxsize/2.
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    s.data['r'] = rr

    vx = s.data['vel'][:,0]
    vy = s.data['vel'][:,1]
    vz = s.data['vel'][:,2]
    s.data['vrad'] = (vx * xx/rr + vy * yy/rr + vz * zz/rr)

    s.data['speed']     = np.linalg.norm(s.data['vel'], axis=1)

    s.data['energdens'] = s.data['u']*s.data['rho']

    # Unit conversion factors from code units to cgs
    # B-field units: Gauss (assuming Gadget/AREPO default cosmological/comoving units check, though here we apply cgs factors for energy density)
    unit_b = np.sqrt(unit_rho * unit_v**2)

    if 'bfld' in s.data:
        # B-field specific energy (energy per unit mass in code units -> cgs)
        b_cgs = s.data['bfld'] * unit_b
        rho_cgs = s.data['rho'] * unit_rho
        s.data['bflden'] = np.sum(b_cgs**2, axis=1) / (8 * np.pi * rho_cgs)
        s.data['bflds'] = np.linalg.norm(s.data['bfld'], axis=1)
        s.data['xb'] = (s.data['bflden'] * rho_cgs)/ (s.data['pres']*(unit_b)**2)
    if 'cren' in s.data:
        s.data['crpres'] = calc_CRP(s.data)
        s.data['xcr'] = s.data['crpres'] / s.data['pres']
        s.data['crendens'] = s.data['cren'] * s.data['rho']
        s.data['ecth'] = s.data['cren'] / s.data['u']
        
    

    if 'coor' in s.data:
        s.data['cool_time'], s.data['cool_rate_mass'], s.data['cool_rate_volume'] = calc_cooling(s.data, s.header)
    else:
        print('\t>>Field COOR not found, skipping cooling calculation...')
 
    if 'ne' in s.data:
        s.data['ne_cm'] = s.data['ne'] * s.data['nH_cm']
  
        s.data['xray'] = calc_bremsstrahlung(s.data)

    return s

def radial_profile_lin(s, field, r_range=(2.5, 200), nbins=200, post_shock=False, shock_path=None, unit_rho=1):
    """Linearly binned radial profile.  Returns (r_centres, profile) or (None, None)."""
    if field not in s.data:
        return None, None
    if post_shock:
        with h5py.File(shock_path, "r") as shocks_file:
            s.data['shocks_coords']     = shocks_file["Coordinates"][:]
            s.data['temperature']       = shocks_file["Temperature"][:]
            s.data['preshock_temp']     = shocks_file["PreShockTemperature"][:]
            s.data['mach']              = shocks_file["Machnumber"][:]
            s.data['shock_direction']   = shocks_file["ShockDirection"][:]
            s.data['preshock_rho']      = shocks_file["PreShockDensity"][:] * unit_rho
            s.data['postshock_rho']     = shocks_file["PostShockDensity"][:] * unit_rho
            s.data['preshock_p']        = shocks_file["PreShockPressure"][:]
            s.data['postshock_p']       = shocks_file["PostShockPressure"][:]
            s.data['preshock_v']        = shocks_file["PreShockVelocity"][:]
            s.data['postshock_v']       = shocks_file["PostShockVelocity"][:]
            s.data['surf']              = shocks_file["Surface"][:]    
            s.data['uflux']         = shocks_file["GeneratedInternalEnergyFlux"][:]
            s.data['edis']          = s.data['uflux']*s.data['surf'] 
        
        # Find the mapping from shocks_coords to pos
        from scipy.spatial import cKDTree

        # Build a KDTree for fast nearest neighbor lookup
        tree = cKDTree(s.data['shocks_coords'])
        _, indices = tree.query(s.data['pos'])

        # Reorder all shock parameters to match pos ordering
        shock_params = ['temperature', 'preshock_temp', 'mach', 'shock_direction', 
                        'preshock_rho', 'postshock_rho', 'preshock_p', 'postshock_p',
                        'preshock_v', 'postshock_v', 'surf', 'uflux', 'edis']

        for param in shock_params:
            if param in s.data:
                s.data[param] = s.data[param][indices]

        # Now shocks_coords should match pos
        s.data['shocks_coords'] = s.data['shocks_coords'][indices]
    pos  = s.data['pos']
    ctr  = np.array([s.boxsize / 2] * 3)
    r    = np.linalg.norm(pos - ctr, axis=1)
    vals = s.data[field]
    rlo, rhi = r_range
    mask = (r >= rlo) & (r <= rhi) & np.isfinite(vals)
    r_bins  = np.linspace(rlo, rhi, nbins + 1)
    r_ctrs  = 0.5 * (r_bins[:-1] + r_bins[1:])
    sums, _ = np.histogram(r[mask], bins=r_bins, weights=vals[mask])
    cnts, _ = np.histogram(r[mask], bins=r_bins)
    with np.errstate(invalid='ignore'):
        profile = np.where(cnts > 0, sums / cnts, np.nan)
    return r_ctrs, profile

def radial_profile_log(s, field, r_range=(2.5, 200), nbins=200):
    """
    Logarithmically binned radial profile.  Returns (r_centres, values).
    Mirrors rvsval() from shocks1d.ipynb.
    """
    pos = s.data['pos']
    ctr = np.array([s.boxsize / 2] * 3)
    r   = np.linalg.norm(pos - ctr, axis=1)

    # radial velocity (attach once)
    if 'vrad' not in s.data:
        diff = pos - ctr
        vdot = np.sum(diff * s.data['vel'], axis=1)
        s.data['vrad'] = vdot / (r + 1e-30)

    if field not in s.data:
        return None, None

    vals     = s.data[field]
    rlo, rhi = r_range
    mask     = (r >= rlo) & (r <= rhi) & np.isfinite(vals)

    r_bins   = np.logspace(np.log10(rlo), np.log10(rhi), nbins + 1)
    r_ctrs   = 0.5 * (r_bins[:-1] + r_bins[1:])
    idx      = np.digitize(r[mask], r_bins)
    profile  = np.array([
        vals[mask][idx == i].mean() if np.any(idx == i) else np.nan
        for i in range(1, nbins + 1)
    ])
    return r_ctrs, profile


def plot_quad_axis(
        s,
        fig,
        quad_subs,
        quad_ax_loc = [0,0],
        var = 'temp',
        weighted = 'rho', # or None
        ranges = None,
        cmap = 'viridis',
        logplot = True,
        divzero = False,
        divzero_centre = None,

        image_proj = 'side',
        proj_on = True,
        proj_fact = 0.1,
        res = 258,
        plotsize = 2,
        colorbar = True,
        add_vec = False,
        vec_val = 'bfld',
        numthreads = 1
        ):


    ## Set up subplot
    quad_ax = plt.Subplot(fig, quad_subs[quad_ax_loc[0],quad_ax_loc[1]])
    fig.add_subplot(quad_ax)

    ## Calculate centre and set up viewing angle
    x_pol = -1 + 2*quad_ax_loc[1]
    y_pol = 1 - 2*quad_ax_loc[0]

    x_centre = s.boxsize/2 + x_pol*plotsize/2
    y_centre = s.boxsize/2 + y_pol*plotsize/2
    z_centre = s.boxsize/2

    if image_proj == 'side':
        plot_centre = [x_centre, z_centre, y_centre]
        axes_sum = [0,2]
    elif image_proj == 'top':
        plot_centre = [x_centre, y_centre, z_centre]
        axes_sum = [0,1]

    print(plot_centre)


    ## Call gadget_snap from arepo-snap-utils to plot in given axis

    if weighted is None:
        s.axplot_Aslice(quad_ax,value=var,cmap=cmap,colorbar=colorbar,divzero=divzero,divzero_centre=divzero_centre,vrange=ranges,axes=axes_sum,logplot=logplot,box=[plotsize,plotsize],center=plot_centre,proj=proj_on,proj_fact=proj_fact,res=res, numthreads=numthreads)
    else:
        s.axplot_Aweightedslice(quad_ax,value=var,weights=weighted,cmap=cmap,colorbar=colorbar,divzero=divzero,divzero_centre=divzero_centre,vrange=ranges,axes=axes_sum,logplot=logplot,box=[plotsize,plotsize],center=plot_centre,proj=proj_on,proj_fact=proj_fact,res=res, numthreads=numthreads)

    ## Set axis labels relative to centre of box

    quad_ax.set_xticklabels(np.round(quad_ax.get_xticks()-s.boxsize/2, decimals=2))
    quad_ax.set_yticklabels(np.round(quad_ax.get_yticks()-s.boxsize/2, decimals=2))

    ## Tidy up axis edges and labels

    if quad_ax_loc[0] == 0:
        quad_ax.spines['bottom'].set_visible(False)
        quad_ax.set_xticks([])
    if quad_ax_loc[0] == 1:
        quad_ax.spines['top'].set_visible(False)
    if quad_ax_loc[1] == 1:
        quad_ax.spines['left'].set_visible(False)
        quad_ax.set_yticks([])

    # Get the last QuadMesh (pcolormesh) from the axis - this is the mappable
    mappable = quad_ax.collections[-1] if quad_ax.collections else None
    
    if add_vec:
        # Add magnetic field streamlines from a simple gridded in-slice field.
        # half = plotsize / 2
        # x_min = plot_centre[axes_sum[0]] - half
        # x_max = plot_centre[axes_sum[0]] + half
        # y_min = plot_centre[axes_sum[1]] - half
        # y_max = plot_centre[axes_sum[1]] + half
        # # x_min, x_max = quad_ax.get_xlim()
        # # y_min, y_max = quad_ax.get_ylim()

        # # Use the full LOS within this panel window to avoid sparse streamline coverage.
        # mask_x = (s.pos[:, axes_sum[0]] >= x_min) & (s.pos[:, axes_sum[0]] <= x_max)
        # mask_y = (s.pos[:, axes_sum[1]] >= y_min) & (s.pos[:, axes_sum[1]] <= y_max)
        # mask = mask_x & mask_y

        # if np.sum(mask) > 10:
        #     pos = s.pos[mask]
        #     vec = s.data[vec_val][mask]

        #     x = pos[:, axes_sum[0]]
        #     y = pos[:, axes_sum[1]]
        #     u = vec[:, axes_sum[0]]
        #     v = vec[:, axes_sum[1]]

        #     x_grid = np.linspace(x_min, x_max, 32)
        #     y_grid = np.linspace(y_min, y_max, 32)
        #     X, Y = np.meshgrid(x_grid, y_grid)

        #     from scipy.interpolate import griddata

        #     # Interpolate scattered particle data directly onto a regular grid
        #     U = griddata((x, y), u, (X, Y), method='nearest')
        #     V = griddata((x, y), v, (X, Y), method='nearest')
        # Add magnetic field streamlines from a gridded in-slice field.
        out_axis = [i for i in [0,1,2] if i not in axes_sum][0]
        slice_val = plot_centre[out_axis]
        dz = plotsize * 0.05  # 5% thickness for selecting particles near the slice

        # Select particles in the slice and within the quadrant bounding box
        mask_out = np.abs(s.pos[:, out_axis] - slice_val) < dz
        mask_x = np.abs(s.pos[:, axes_sum[0]] - plot_centre[axes_sum[0]]) < (plotsize/2)
        mask_y = np.abs(s.pos[:, axes_sum[1]] - plot_centre[axes_sum[1]]) < (plotsize/2)
        mask = mask_out & mask_x & mask_y


        # Subsample particles further to avoid cluttered quiver plot
        if np.sum(mask) > 0:
            sample_step = max(1, int(np.sum(mask) / 500))  # Target ~500 vectors
            sampled_pos = s.pos[mask][::sample_step]
            sampled_B = s.data[vec_val][mask][::sample_step]

            x_pos = sampled_pos[:, axes_sum[0]]
            y_pos = sampled_pos[:, axes_sum[1]]
            u_v = sampled_B[:, axes_sum[0]]
            v_v = sampled_B[:, axes_sum[1]]

            quad_ax.quiver(x_pos, y_pos, u_v, v_v, color='gray',
                        alpha=0.8, scale=None, pivot='mid', headwidth=3, headlength=4)

            
            
            pos = s.pos#[mask]
            vec = s.data[vec_val]#[mask]

            x = pos[:, axes_sum[0]]
            y = pos[:, axes_sum[1]] 
            u = vec[:, axes_sum[0]]
            v = vec[:, axes_sum[1]]

            half = plotsize / 2.0
            x_min = plot_centre[axes_sum[0]] - half
            x_max = plot_centre[axes_sum[0]] + half
            y_min = plot_centre[axes_sum[1]] - half
            y_max = plot_centre[axes_sum[1]] + half
            print(f"Streamline grid bounds: x=({x_min:.2f}, {x_max:.2f}), y=({y_min:.2f}, {y_max:.2f})")
            x_grid = np.linspace(x_min, x_max, 256)
            y_grid = np.linspace(y_min, y_max, 256)
            X, Y = np.meshgrid(x_grid, y_grid)

            from scipy.interpolate import griddata
            import lic

            U = griddata((x, y), u, (X, Y), method='nearest')
            V = griddata((x, y), v, (X, Y), method='nearest')

            # Don't zero out NaNs - use nearest neighbour so there shouldn't be any
            # but just in case:
            # U = np.where(np.isnan(U), np.nanmean(u), U)
            # V = np.where(np.isnan(V), np.nanmean(v), V)

            xlim = quad_ax.get_xlim()
            ylim = quad_ax.get_ylim()

            # quad_ax.streamplot(X, Y, U, V, 
            #             color='white',
            #     density=0.5,
            #     linewidth=0.5,
            #     arrowsize=0.7,
            #     zorder=6)
            
            print(quad_ax.get_xlim(), quad_ax.get_ylim())
            lic_img = lic.lic(U.T, V.T, length=50)
            
            quad_ax.imshow(
                lic_img.T,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap='gray',
                alpha=0.3,
                zorder=6,
                aspect='auto'
            )

            quad_ax.set_xlim(xlim)
            quad_ax.set_ylim(ylim)
    return quad_ax, mappable

def find_shock_radius(s, r_range=(1e-3,1), nbins=500):
    # r_shock, t_arr = [], []
    # for n, (s, t) in enumerate(zip(snaps, times)):
    r, mach = radial_profile_lin(s, 'mach', r_range=r_range, nbins=nbins)
    idx = np.nanargmax(mach)
    r_shock = r[idx]
    
    # Find second maximum (reverse shock)
    mach_copy = mach.copy()
    right, left =0, 0
    while mach_copy[idx+right] > 0 and idx+right < len(mach_copy)-1:
        right += 1  
    while mach_copy[idx-left] > 0 and idx-left > 0:
        left += 1
    mach_copy[max(0, idx-left):min(len(mach_copy), idx+right+1)] = np.nan  # Exclude the forward shock region
    idx_reverse = np.nanargmax(mach_copy)
    r_reverse_shock = r[idx_reverse]
    
    return r_shock, r_reverse_shock